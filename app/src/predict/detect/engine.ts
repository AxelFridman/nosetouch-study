/**
 * The live session engine (types.ts section 6, APP_SPEC 4.2): cue-free and continuous.
 *
 *   push(samples) -> LiveBuffer; whenever stream time crosses the next tickMs (100 ms) boundary, one tick:
 *     stream = pipeline.streamFrom(buffer.raw()) -> activity -> bouts (liveBouts)
 *     every bout that became final under the freeze rule (ended, look-ahead in the buffer, the stream extends
 *     settleMs = 1 s past its decision) is processed ONCE and frozen: its window, features, P(class) and the attempt
 *     rule; an attempt gets every bundled model's output (AttemptResult), anything else is an IgnoredMovement;
 *     the grip check (rest_up_z, lane m4's gate) is applied when the first bout is frozen;
 *     after each new attempt the driver's stopping rule runs (streamEnded false): confident / k_max -> final;
 *     at maxStreamMs (180 s) the session ends as stream_end (attempts) or no_attempt.
 *   final / grip: samples are ignored until reset(); setDriver recomputes from the frozen attempts at once.
 *
 * Cost (phones): a tick reuses the previous tick's activity and median grid step (LiveDetectCache, exact), computes
 * the rest pose only when a bout is about to be frozen or the gate is still provisional, and, with deferScoringMs,
 * runs the four models of a new attempt (the CNN is most of a freeze) only once the phone is still again or that
 * long after the attempt was frozen. The window is cut at the freeze tick either way, so every number is the same;
 * only the moment the attempt appears (and a decision is taken) moves, out of the next movement.
 *
 * Driven only by sample times (never by wall-clock timers): a replay of the same samples gives the same snapshots.
 * After the 1 s settle margin the zero-phase low-pass no longer changes anything a frozen bout depends on, so the
 * frozen values equal predict_stream's on the complete recording (PORT_SPEC 2.15).
 *
 * The detector, the attempt rule, the grip gate and the stopping thresholds (p_stop 0.9, k_max 5, p_answer 0.5) are
 * lane m4's, from the exported bundle: they were fitted for lane m4's own classifier and are applied unchanged to
 * every model. Lane prod, which would refit them for the final CNN, has not run.
 */
import type {
  ApproachWindow,
  AttemptResult,
  Bout,
  CreateSessionEngine,
  Decision,
  DetectorBundle,
  EngineEvent,
  EnginePhase,
  EngineSnapshot,
  HandModel,
  IgnoredMovement,
  LiveBuffer,
  LiveSample,
  ModelAttemptOutput,
  ModelComparison,
  ModelId,
  PipelineApi,
  SessionEngine,
  SessionEngineOptions,
  Stream,
} from '../types';
import type { WindowPipeline } from './canonical';
import { ignoredReason } from './classifier';
import { gripBlocked, restDetails, restIsFinal } from './grip';
import { LiveDetectCache, liveBouts, sameBout, screenAwayHint } from './live';
import { segmentRow, type SegmentRowDetail } from './rows';
import { decisionOrder, gripDecision, isAvailable, percentRight, stoppingRule } from './stopping';

export const DEFAULT_TICK_MS = 100;
export const DEFAULT_SETTLE_MS = 1000;
export const DEFAULT_MAX_STREAM_MS = 180_000;
const RATE_WINDOW_MS = 2000;

/** What the engine needs besides SessionEngineOptions: pipeline/ and lane m4's detector bundle. */
export interface EngineDeps {
  readonly pipeline: Pick<PipelineApi, 'createLiveBuffer' | 'streamFrom'> & WindowPipeline;
  readonly detector: DetectorBundle;
}

/** createSessionEngine bound to the pipeline and the detector (models/registry.ts detector()). */
export function makeCreateSessionEngine(deps: EngineDeps): CreateSessionEngine {
  return (options: SessionEngineOptions): SessionEngine => new Engine(deps, options);
}

/** An attempt whose window is cut (at its freeze tick) and whose models have not run yet (deferScoringMs). */
interface PendingAttempt {
  readonly bout: Bout;
  readonly row: SegmentRowDetail;
  readonly window: ApproachWindow;
  readonly frozenAtMs: number;
}

/** A model that cannot score a window (it threw) is unavailable for that attempt. */
function safePredict(m: HandModel, window: ApproachWindow): ModelAttemptOutput {
  try {
    return m.predict({ window });
  } catch {
    return { modelId: m.spec.id, zRaw: NaN, pRight: NaN, z: NaN, available: false } as ModelAttemptOutput;
  }
}

class Engine implements SessionEngine {
  private readonly buffer: LiveBuffer;
  private readonly tickMs: number;
  private readonly settleMs: number;
  private readonly maxStreamMs: number;
  private readonly deferScoringMs: number;
  private readonly cache: LiveDetectCache;
  private pending: PendingAttempt[] = [];
  private driver: ModelId;
  private phase: EnginePhase = 'watching';
  private attempts: AttemptResult[] = [];
  private ignored: IgnoredMovement[] = [];
  private frozen: Bout[] = [];
  private restUpZ: number | null = null;
  private restFinal = false;
  private nextTick: number;
  private tMs = 0;
  private times: number[] = [];
  private still = false;
  private hint = false;
  private lastEvent: EngineEvent | null = null;
  private decision: Decision;
  private snap: EngineSnapshot;
  private readonly listeners = new Set<(s: EngineSnapshot) => void>();

  constructor(
    private readonly deps: EngineDeps,
    private readonly options: SessionEngineOptions,
  ) {
    if (!options.models.some((m) => m.spec.id === options.driver)) throw new Error(`engine: no model '${options.driver}'`);
    this.buffer = deps.pipeline.createLiveBuffer();
    this.tickMs = options.tickMs ?? DEFAULT_TICK_MS;
    this.settleMs = options.settleMs ?? DEFAULT_SETTLE_MS;
    this.maxStreamMs = options.maxStreamMs ?? DEFAULT_MAX_STREAM_MS;
    this.deferScoringMs = options.deferScoringMs ?? 0;
    this.cache = new LiveDetectCache(deps.detector);
    this.driver = options.driver;
    this.nextTick = this.tickMs;
    this.decision = this.decide(false);
    this.snap = this.build();
  }

  // ───────────── SessionEngine ─────────────

  push(samples: LiveSample | readonly LiveSample[]): void {
    if (this.phase !== 'watching') return; // final / grip: wait for reset(); stopped: disposed
    const list = Array.isArray(samples) ? (samples as readonly LiveSample[]) : [samples as LiveSample];
    let changed = false;
    for (const s of list) {
      if (this.phase !== 'watching') break;
      if (this.buffer.push(s) < 0) continue;
      const t0 = this.buffer.t0 ?? s.t;
      this.tMs = s.t - t0;
      this.times.push(this.tMs);
      changed = true;
      if (this.tMs >= this.nextTick) {
        this.nextTick = (Math.floor(this.tMs / this.tickMs) + 1) * this.tickMs;
        this.tick();
      }
    }
    if (changed) this.emit();
  }

  setDriver(id: ModelId): void {
    if (!this.options.models.some((m) => m.spec.id === id)) throw new Error(`engine: no model '${id}'`);
    this.driver = id;
    if (this.phase === 'grip') this.decision = gripDecision(id);
    else if (this.phase !== 'stopped') {
      const wasFinal = this.phase === 'final';
      this.decision = this.decide(wasFinal);
      if (this.decision.final && !wasFinal) this.finish();
    }
    this.emit();
  }

  reset(): void {
    if (this.phase === 'stopped') return;
    this.clear();
    this.lastEvent = { type: 'reset', atMs: 0 };
    this.emit();
  }

  notifyGap(): void {
    if (this.phase !== 'watching') return; // a final answer (or the grip message) stays
    const at = this.tMs;
    this.clear();
    this.lastEvent = { type: 'gap', atMs: at };
    this.emit();
  }

  snapshot(): EngineSnapshot {
    return this.snap;
  }

  subscribe(listener: (s: EngineSnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  dispose(): void {
    if (this.phase === 'stopped') return;
    this.phase = 'stopped';
    this.emit();
    this.listeners.clear();
    this.buffer.clear();
  }

  // ───────────── internals ─────────────

  private clear(): void {
    this.buffer.clear();
    this.phase = 'watching';
    this.attempts = [];
    this.ignored = [];
    this.frozen = [];
    this.pending = [];
    this.cache.reset();
    this.restUpZ = null;
    this.restFinal = false;
    this.nextTick = this.tickMs;
    this.tMs = 0;
    this.times = [];
    this.still = false;
    this.hint = false;
    this.decision = this.decide(false);
  }

  private tick(): void {
    const { pipeline, detector } = this.deps;
    let stream: Stream;
    try {
      stream = pipeline.streamFrom(this.buffer.raw(), this.options.normalization);
    } catch (e) {
      if (e instanceof RangeError) return; // not enough data yet (fewer than 5 rows)
      throw e;
    }
    const n = stream.t.length;
    if (n === 0) return;
    const lb = liveBouts(stream, detector, this.settleMs, this.cache);
    this.still = lb.still;
    const isNew = (b: Bout): boolean => !this.frozen.some((f) => sameBout(f, b));
    if (!this.restFinal && (this.frozen.length > 0 || lb.bouts.some((b, k) => lb.frozen[k] && isNew(b)))) {
      // rest_up_z reads the rest before the first DEFAULT bout: final once that bout is frozen (restIsFinal). It is
      // only read when a bout is frozen (the gate below, and the snapshot), so it is computed only then.
      const r = restDetails(stream, detector, { act: lb.activity, dtMs: lb.dtMs });
      this.restUpZ = Number.isNaN(r.restUpZ) ? null : r.restUpZ;
      this.restFinal = restIsFinal(r, stream, this.settleMs);
    }
    // lane m4's grip gate (the rest pose before the first movement has the screen facing away or down), applied as
    // soon as a bout is frozen, and again on later ticks while the rest pose is still provisional
    const gripNow = (): boolean => gripBlocked(this.restUpZ, detector);
    if (this.frozen.length > 0 && gripNow()) this.blockGrip();
    for (let k = 0; k < lb.bouts.length && this.phase === 'watching'; k++) {
      const b = lb.bouts[k];
      if (!lb.frozen[k] || !isNew(b)) continue;
      if (gripNow()) {
        this.blockGrip();
        break;
      }
      this.freeze(stream, b, k > 0 ? lb.bouts[k - 1] : null);
    }
    this.hint = this.phase === 'watching' && screenAwayHint(stream, lb.activity, detector);
    const capped = this.tMs >= this.maxStreamMs;
    if (this.phase === 'watching' && this.pending.length > 0) {
      if (capped || lb.still || this.tMs - this.pending[0].frozenAtMs >= this.deferScoringMs) this.scorePending();
    }
    if (this.phase === 'watching' && capped) {
      this.decision = this.decide(true);
      this.finish();
    }
  }

  /** the models of the waiting attempts, in order, until a decision is final */
  private scorePending(): void {
    const list = this.pending;
    this.pending = [];
    for (const p of list) {
      if (this.phase !== 'watching') break;
      this.score(p);
    }
  }

  private freeze(stream: Stream, b: Bout, prev: Bout | null): void {
    const { pipeline, detector } = this.deps;
    const row = segmentRow(stream, b, detector, pipeline, prev);
    this.frozen.push(b);
    if (row.isAttempt && row.window !== null) {
      const p: PendingAttempt = { bout: b, row, window: row.window, frozenAtMs: this.tMs };
      if (this.deferScoringMs > 0 || this.pending.length > 0) this.pending.push(p);
      else this.score(p);
    } else {
      const reason = ignoredReason(row.cls, row.proba, row.canonicalOk, detector.attemptRule) ?? 'other';
      this.ignored = [...this.ignored, { bout: b, cls: row.cls, pApproach: row.proba.approach, reason, frozenAtMs: this.tMs }];
      this.lastEvent = { type: 'ignored', reason, atMs: this.tMs };
    }
  }

  /** every bundled model's output for the attempt's window, then the driver's stopping rule */
  private score(p: PendingAttempt): void {
    const { bout: b, row, window } = p;
    const outputs = Object.fromEntries(this.options.models.map((m) => [m.spec.id, safePredict(m, window)])) as Record<
      ModelId,
      ModelAttemptOutput
    >;
    const k = this.attempts.length + 1;
    this.attempts = [
      ...this.attempts,
      {
        k,
        bout: b,
        pApproach: row.proba.approach,
        pReturn: row.proba.return,
        pOther: row.proba.other,
        closureDeg: row.closureDeg,
        restUpZBefore: NaN, // final_robust.rest_up_z is not computed on the page (QC only)
        outputs,
        frozenAtMs: this.tMs,
      },
    ];
    this.lastEvent = { type: 'attempt', k, atMs: this.tMs };
    this.decision = this.decide(false);
    if (this.decision.final) this.finish();
  }

  private finish(): void {
    this.phase = 'final';
    this.lastEvent = { type: 'final', atMs: this.tMs };
  }

  private blockGrip(): void {
    this.phase = 'grip';
    this.decision = gripDecision(this.driver);
    this.lastEvent = { type: 'final', atMs: this.tMs };
  }

  private model(id: ModelId): HandModel {
    const m = this.options.models.find((x) => x.spec.id === id);
    if (!m) throw new Error(`engine: no model '${id}'`);
    return m;
  }

  private decide(streamEnded: boolean): Decision {
    return stoppingRule(this.model(this.driver), this.attempts, this.deps.detector.stopping, streamEnded);
  }

  /** every model after the same attempts the driver used (its decision time), each with its own aggregation */
  private compare(): ModelComparison[] {
    const { pStop, kMax } = this.deps.detector.stopping;
    const until = this.phase === 'grip' ? -Infinity : (this.decision.decidedMs ?? -Infinity);
    const ordered = decisionOrder(this.attempts.map((a) => ({ decidedMs: a.bout.decidedMs, seg: a.bout.index, a })));
    const used = ordered.filter((x) => x.decidedMs <= until).map((x) => x.a);
    return this.options.models.map((m) => {
      const outs = used.map((a) => a.outputs[m.spec.id]).filter(isAvailable).slice(0, kMax);
      const p = outs.length ? m.aggregate(outs) : 0.5;
      return { modelId: m.spec.id, pRight: p, percentRight: percentRight(p), k: outs.length, confident: Math.max(p, 1 - p) >= pStop };
    });
  }

  private rateHz(): number | null {
    const ts = this.times;
    if (ts.length < 2) return null;
    const last = ts[ts.length - 1];
    let i = ts.length - 1;
    while (i > 0 && ts[i - 1] >= last - RATE_WINDOW_MS) i--;
    const span = last - ts[i];
    return span > 0 ? ((ts.length - 1 - i) * 1000) / span : null;
  }

  private build(): EngineSnapshot {
    return {
      phase: this.phase,
      tMs: this.tMs,
      nSamples: this.buffer.n,
      rateHz: this.rateHz(),
      still: this.still,
      screenAwayHint: this.hint,
      // known once a bout is frozen, or when the grip gate fired on the first one (it blocks before freezing it)
      restUpZ: this.frozen.length > 0 || this.phase === 'grip' ? this.restUpZ : null,
      attempts: this.attempts,
      ignored: this.ignored,
      driver: this.driver,
      decision: this.decision,
      compare: this.compare(),
      stopping: this.deps.detector.stopping,
      lastEvent: this.lastEvent,
    };
  }

  private emit(): void {
    this.snap = this.build();
    for (const l of [...this.listeners]) l(this.snap);
  }
}
