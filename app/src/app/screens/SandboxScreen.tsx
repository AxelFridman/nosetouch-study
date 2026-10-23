/**
 * Researcher sandbox (nothing is saved): live sensor signals and the online movement detector.
 *  - Start/Stop a source from services.createSource() (real sensors, or the simulator with ?sim=1).
 *  - Rolling 5 s plots: the detector's smoothed |ω| (with the still/onset thresholds) and smoothed
 *    |a_lin| plus the gravity components (low-passed accelerationIncludingGravity).
 *  - SandboxDetector runs an OnlineSegmenter with an automatic GO once the phone has been still for
 *    600 ms; every detected movement gets GO/onset/end markers and a duration, then the detector
 *    re-arms (after post_roll_ms, or at once after a failure).
 *  - Shows the measured sample rate over the visible window.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type {
  CueKind,
  MotionSample,
  PermissionState,
  SegmentationParams,
  SegmentationResult,
  SensorSource,
  Vec3,
} from '../../core/types';
import { DEFAULT_SEGMENTATION_PARAMS } from '../../core/constants';
import {
  OnlineSegmenter,
  type RelMotionSample,
  type SegmenterEvent,
  type SegmenterLive,
  type SegmenterState,
} from '../../core/segmentation';
import { requestMotionPermission } from '../../core/sensors';
import { Button, Card, Notice, Screen, Stat } from '../../ui/components';
import { SignalPlot, type AccSource, type SignalMarker } from '../../ui/SignalPlot';
import { useStrings } from '../../i18n';
import { getServices } from '../services';
import { getState, goBack, setState } from '../store';
import '../../ui/session.css';

const WINDOW_MS = 5000;
const GO_DELAY_MS = 600;
const GRAVITY_DISPLAY_TAU_MS = 200;
const MAX_KEPT = 50;

// ───────────────────────────── detector (pure, testable) ─────────────────────────────

/** The part of OnlineSegmenter the sandbox uses (lets tests inject a fake). */
export interface SegmenterLike {
  push(s: RelMotionSample): SegmenterEvent[];
  markGo(t: number): void;
  readonly state: SegmenterState;
  readonly live: SegmenterLive;
  result(): SegmentationResult;
}

export interface SandboxMovement {
  n: number;
  go_t: number;
  onset_t: number | null;
  end_t: number | null;
  duration_ms: number | null;
  peak_gyro_dps: number | null;
  outcome: 'pending' | 'complete' | 'no_movement' | 'too_long';
}

export interface SandboxMarker {
  t: number;
  kind: 'go' | 'onset' | 'end';
}

export interface SandboxDetectorOptions {
  goDelayMs?: number;
  makeSegmenter?: (p: SegmentationParams) => SegmenterLike;
  /** called when the sandbox would play a cue; t is relative to the start of the run */
  onCue?: (kind: CueKind, t: number) => void;
}

/**
 * Online segmenter with an automatic GO (stillness + goDelayMs) that re-arms after every movement.
 * Feed samples with times relative to the start of the run (strictly increasing).
 */
export class SandboxDetector {
  readonly movements: SandboxMovement[] = [];
  readonly markers: SandboxMarker[] = [];
  private seg: SegmenterLike;
  private index = 0;
  private goAt: number | null = null;
  private rearmAt: number | null = null;
  private current: SandboxMovement | null = null;
  private count = 0;

  constructor(
    private readonly params: SegmentationParams,
    private readonly opts: SandboxDetectorOptions = {},
  ) {
    this.seg = this.make();
  }

  private make(): SegmenterLike {
    return this.opts.makeSegmenter ? this.opts.makeSegmenter(this.params) : new OnlineSegmenter(this.params);
  }

  get state(): SegmenterState {
    return this.seg.state;
  }

  get live(): SegmenterLive {
    return this.seg.live;
  }

  push(sample: MotionSample): SegmenterEvent[] {
    if (this.rearmAt !== null && sample.t >= this.rearmAt) {
      this.seg = this.make();
      this.index = 0;
      this.rearmAt = null;
      this.goAt = null;
      this.current = null;
    }
    const events = this.seg.push({ ...sample, index: this.index++ });
    for (const e of events) this.handle(e, sample.t);
    // GO right after the push of the sample that reached the GO time (as the trial controller does)
    if (this.goAt !== null && sample.t >= this.goAt && this.seg.state === 'still') {
      this.goAt = null;
      this.seg.markGo(sample.t);
      this.count++;
      this.current = {
        n: this.count,
        go_t: sample.t,
        onset_t: null,
        end_t: null,
        duration_ms: null,
        peak_gyro_dps: null,
        outcome: 'pending',
      };
      this.keep(this.movements, this.current);
      this.keep(this.markers, { t: sample.t, kind: 'go' });
      this.opts.onCue?.('go', sample.t);
    }
    return events;
  }

  private keep<T>(list: T[], item: T): void {
    list.push(item);
    if (list.length > MAX_KEPT) list.splice(0, list.length - MAX_KEPT);
  }

  private handle(e: SegmenterEvent, t: number): void {
    switch (e.type) {
      case 'still':
        this.goAt = e.t + (this.opts.goDelayMs ?? GO_DELAY_MS);
        break;
      case 'still_lost':
      case 'false_start':
        this.goAt = null;
        break;
      case 'onset':
        if (this.current) this.current.onset_t = e.t;
        this.keep(this.markers, { t: e.t, kind: 'onset' });
        break;
      case 'end': {
        const r = this.seg.result();
        if (this.current) {
          this.current.end_t = e.t;
          this.current.duration_ms =
            r.duration_ms ?? (this.current.onset_t !== null ? e.t - this.current.onset_t : null);
          this.current.peak_gyro_dps = r.peak_gyro_dps;
          this.current.outcome = 'complete';
        }
        this.keep(this.markers, { t: e.t, kind: 'end' });
        this.opts.onCue?.('end', t);
        this.rearmAt = e.detected_t + this.params.post_roll_ms;
        break;
      }
      case 'no_movement':
      case 'too_long':
        if (this.current) this.current.outcome = e.type;
        this.opts.onCue?.('error', t);
        this.rearmAt = t; // a fresh detector from the next sample on
        break;
    }
  }
}

// ───────────────────────────── screen ─────────────────────────────

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isVec = (v: Vec3 | null): v is Vec3 => !!v && isNum(v[0]) && isNum(v[1]) && isNum(v[2]);

interface Frame {
  t: number[];
  gyro: number[];
  acc: number[];
  gx: number[];
  gy: number[];
  gz: number[];
  xRange: [number, number];
  markers: SandboxMarker[];
  rateHz: number | null;
  samples: number;
  state: SegmenterState;
  accSource: AccSource | null;
  movements: SandboxMovement[];
}

interface SandboxRun {
  source: SensorSource;
  t0: number;
  detector: SandboxDetector;
  cols: { t: number[]; gyro: number[]; acc: number[]; gx: number[]; gy: number[]; gz: number[] };
  n: number;
  grav: Vec3 | null;
  gravT: number;
  accSource: AccSource | null;
  ticker: ReturnType<typeof setInterval> | undefined;
}

function onSample(run: SandboxRun, m: MotionSample): void {
  const t = m.t - run.t0;
  const cols = run.cols;
  const last = cols.t.length ? cols.t[cols.t.length - 1] : -Infinity;
  if (!(t > last)) return; // the segmenter needs strictly increasing times
  run.n++;
  if (run.accSource === null) run.accSource = isVec(m.acc) ? 'linear' : isVec(m.accg) ? 'highpass' : 'none';
  run.detector.push({ ...m, t });
  const live = run.detector.live;
  if (isVec(m.accg)) {
    if (!run.grav) run.grav = [m.accg[0], m.accg[1], m.accg[2]];
    else {
      const k = 1 - Math.exp(-Math.max(0, t - run.gravT) / GRAVITY_DISPLAY_TAU_MS);
      const g = run.grav;
      run.grav = [g[0] + k * (m.accg[0] - g[0]), g[1] + k * (m.accg[1] - g[1]), g[2] + k * (m.accg[2] - g[2])];
    }
    run.gravT = t;
  }
  cols.t.push(t);
  cols.gyro.push(live.gyro);
  cols.acc.push(live.acc);
  cols.gx.push(run.grav ? run.grav[0] : NaN);
  cols.gy.push(run.grav ? run.grav[1] : NaN);
  cols.gz.push(run.grav ? run.grav[2] : NaN);
  // drop what scrolled out of the window (in chunks, to keep pushes O(1) amortized)
  const cut = t - WINDOW_MS - 250;
  if (cols.t.length > 64 && cols.t[32] < cut) {
    let k = 0;
    while (k < cols.t.length && cols.t[k] < cut) k++;
    for (const key of ['t', 'gyro', 'acc', 'gx', 'gy', 'gz'] as const) cols[key].splice(0, k);
  }
}

function snapshot(run: SandboxRun): Frame {
  const c = run.cols;
  const last = c.t.length ? c.t[c.t.length - 1] : 0;
  const from = last - WINDOW_MS;
  let first = 0;
  while (first < c.t.length && c.t[first] < from) first++;
  const n = c.t.length - first;
  const span = n > 1 ? c.t[c.t.length - 1] - c.t[first] : 0;
  return {
    t: c.t.slice(),
    gyro: c.gyro.slice(),
    acc: c.acc.slice(),
    gx: c.gx.slice(),
    gy: c.gy.slice(),
    gz: c.gz.slice(),
    xRange: [from, last],
    markers: run.detector.markers.filter((m) => m.t >= from - 1000),
    rateHz: span > 0 ? ((n - 1) / span) * 1000 : null,
    samples: run.n,
    state: run.detector.state,
    accSource: run.accSource,
    movements: run.detector.movements.slice(-5).reverse(),
  };
}

const MARKER_COLORS: Record<SandboxMarker['kind'], string> = {
  go: 'var(--phase-go)',
  onset: 'var(--hand-left)',
  end: 'var(--phase-done)',
};

export function SandboxScreen() {
  const s = useStrings();
  const t = s.session.sandbox;
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const runRef = useRef<SandboxRun | null>(null);
  const params = DEFAULT_SEGMENTATION_PARAMS;

  const stop = () => {
    const run = runRef.current;
    if (!run) return;
    runRef.current = null;
    clearInterval(run.ticker);
    try {
      run.source.stop();
    } catch (e) {
      console.warn('stopping the sensor source failed', e);
    }
    setFrame(snapshot(run));
    setRunning(false);
  };

  useEffect(() => () => stop(), []);

  const begin = () => {
    const services = getServices();
    let source: SensorSource;
    try {
      source = services.createSource();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    const t0 = source.clock.now();
    const detector = new SandboxDetector(params, {
      goDelayMs: GO_DELAY_MS,
      onCue: (kind, tRel) => {
        try {
          services.cues.play(kind);
          // lets a simulated participant react exactly as in a trial
          if (kind !== 'error') source.notifyCue?.(kind, tRel + t0);
        } catch {
          // cues are best effort
        }
      },
    });
    const run: SandboxRun = {
      source,
      t0,
      detector,
      cols: { t: [], gyro: [], acc: [], gx: [], gy: [], gz: [] },
      n: 0,
      grav: null,
      gravT: 0,
      accSource: null,
      ticker: undefined,
    };
    runRef.current = run;
    run.ticker = setInterval(() => {
      if (runRef.current === run) setFrame(snapshot(run));
    }, 66);
    setRunning(true);
    source
      .start({
        onMotion: (m) => {
          if (runRef.current === run) onSample(run, m);
        },
        onOrientation: () => undefined,
        onError: (e) => setError(t.errors[e.code] ?? e.message),
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
        stop();
      });
  };

  const start = () => {
    if (runRef.current) return;
    setError(null);
    setFrame(null);
    const st0 = getState();
    const caps = st0.capabilities;
    // iOS: request motion access synchronously inside the tap
    let permission: Promise<PermissionState> | null = null;
    if (!st0.settings.simulate && caps?.permission_api && caps.permission_state !== 'granted') {
      try {
        permission = requestMotionPermission();
      } catch (e) {
        console.warn('motion permission request failed', e);
      }
    }
    try {
      void getServices().cues.unlock().catch(() => undefined);
    } catch {
      // audio is optional
    }
    if (!permission) {
      begin();
      return;
    }
    permission.then(
      (p) => {
        const capsNow = getState().capabilities;
        if (capsNow) setState({ capabilities: { ...capsNow, permission_state: p } });
        if (p === 'denied') setError(t.permissionDenied);
        else begin();
      },
      () => begin(),
    );
  };

  const markers: SignalMarker[] = (frame?.markers ?? []).map((m) => ({
    t: m.t,
    label: t.markers[m.kind],
    color: MARKER_COLORS[m.kind],
  }));
  const state = running ? (frame?.state ?? 'waiting_still') : 'idle';

  return (
    <Screen title={t.title} onBack={goBack} testId="screen-sandbox">
      <Notice kind="info">{t.intro}</Notice>
      {running ? (
        <Button variant="danger" size="lg" block testId="btn-sandbox-stop" onClick={stop}>
          {t.stop}
        </Button>
      ) : (
        <Button size="lg" block testId="btn-sandbox-start" onClick={start}>
          {t.start}
        </Button>
      )}
      {error && <Notice kind="error">{error}</Notice>}

      <Card>
        <div class="sandbox-stats">
          <Stat
            label={t.detector}
            value={
              <span class="sandbox-state" data-state={state} data-testid="sandbox-state">
                {t.states[state]}
              </span>
            }
          />
          <Stat
            label={t.rate}
            value={frame?.rateHz != null ? `${frame.rateHz.toFixed(0)} Hz` : '–'}
            testId="sandbox-rate"
          />
          <Stat label={t.samples} value={frame?.samples ?? 0} />
          <Stat label={t.accSource} value={frame?.accSource ? t.accSources[frame.accSource] : '–'} />
        </div>
      </Card>

      <section class="stack-sm">
        <h3>{t.gyroPlot}</h3>
        <SignalPlot
          t={frame?.t ?? []}
          series={[{ values: frame?.gyro ?? [], color: 'var(--primary)', label: t.series.gyro }]}
          markers={markers}
          hLines={[
            { y: params.still_gyro_dps, label: t.thresholds.still, color: 'var(--ok)' },
            { y: params.onset_gyro_min_dps, label: t.thresholds.onset, color: 'var(--danger)' },
          ]}
          xRange={frame?.xRange}
          height={150}
          testId="sandbox-gyro-plot"
        />
      </section>

      <section class="stack-sm">
        <h3>{t.accPlot}</h3>
        <SignalPlot
          t={frame?.t ?? []}
          series={[
            { values: frame?.acc ?? [], color: 'var(--warn)', label: t.series.acc },
            { values: frame?.gx ?? [], color: '#d946ef', label: t.series.gx },
            { values: frame?.gy ?? [], color: '#16a34a', label: t.series.gy },
            { values: frame?.gz ?? [], color: '#0ea5e9', label: t.series.gz },
          ]}
          markers={markers}
          xRange={frame?.xRange}
          height={170}
          testId="sandbox-acc-plot"
        />
      </section>

      <Card title={t.movements} testId="sandbox-movements">
        {frame && frame.movements.length > 0 ? (
          <ul class="clean sandbox-movements">
            {frame.movements.map((m) => (
              <li key={m.n}>
                {m.outcome === 'complete' && m.duration_ms !== null
                  ? t.movementRow(m.n, Math.round(m.duration_ms), m.peak_gyro_dps !== null ? m.peak_gyro_dps.toFixed(0) : '–')
                  : m.outcome === 'no_movement'
                    ? `#${m.n}: ${t.noMovement}`
                    : m.outcome === 'too_long'
                      ? `#${m.n}: ${t.tooLong}`
                      : `#${m.n}: …`}
              </li>
            ))}
          </ul>
        ) : (
          <p class="muted">{t.noMovements}</p>
        )}
      </Card>
    </Screen>
  );
}
