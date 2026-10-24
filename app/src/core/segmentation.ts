/**
 * Online (causal) movement segmentation: hold-still detection, onset and end of the nose-touch
 * movement. Pure logic, no DOM, no timers — driven only by motion samples, so it runs identically
 * in the browser, in the simulator and in Node tests.
 *
 * ALGORITHM "online_v1" — EXACT spec. analysis/motionlab/segment.py ports it and must give the
 * same indices on the same input; docs/04_SEGMENTATION.md explains it in prose.
 *
 *  Notation: sample i has relative time t_i (ms) and dt_i = t_i − t_{i−1} (dt_0 = 0).
 *  EMA(x; τ): S_0 = x_0; S_i = S_{i−1} + k_i·(x_i − S_{i−1}), k_i = 1 − exp(−dt_i/τ) (τ = 0 → S_i = x_i).
 *  "C holds continuously for D ms" at sample i: C held at every sample of the current run i0..i
 *  (i0 = first sample of the run) and t_i − t_{i0} ≥ D. A sample where C is false ends the run.
 *
 *  1. Raw signals:
 *       gyro_i = sqrt(α²+β²+γ²) of rr; if rr is null (or any component non-finite) → reuse the
 *                previous raw gyro value (0 if none yet).
 *       acc_source is decided by the FIRST sample: acc present (3 finite numbers) → 'linear';
 *                else accg present → 'highpass'; else 'none'.
 *       'linear':   a_i = |acc_i| (missing in a later sample → reuse the previous raw value)
 *       'highpass': G = EMA(accg; gravity_tau_ms) per component (G_0 = accg_0), a_i = |accg_i − G_i|
 *                   (G updated with sample i BEFORE subtracting; missing accg → reuse previous a)
 *       'none':     a_i = 0
 *     Smoothed: gyro_s = EMA(gyro; smoothing_ms), acc_s = EMA(a; smoothing_ms).
 *     C_still(i) := gyro_s < still_gyro_dps AND acc_s < still_acc_ms2.
 *  2. WAITING_STILL: when C_still holds continuously for still_hold_ms → emit {type:'still', t: t_i}
 *     → STILL. stillProgress = C_still(i) ? min(1, (t_i − t_{i0}) / still_hold_ms) : 0.
 *     The baseline buffer = every sample of the current still run (from i0), kept while STILL.
 *  3. STILL (before GO): each sample with C_still true is appended to the baseline buffer.
 *     A sample with C_still false: if gyro_s > onset_gyro_min_dps OR acc_s > onset_acc_min_ms2 →
 *     emit 'false_start', else emit 'still_lost'; in both cases → WAITING_STILL, buffer cleared, the
 *     still run restarts at the next sample where C_still holds.
 *  4. markGo(t) (called by the controller right after push() of the sample at which GO was given):
 *     baseline = { n, gyro_mean, gyro_sd, acc_mean, acc_sd (population sd, ddof 0, of the SMOOTHED
 *     values in the buffer), gravity = mean raw accg of the buffer (null if no accg) } → ARMED.
 *       thr_gyro = max(onset_gyro_min_dps, gyro_mean + onset_k_sd·gyro_sd)
 *       thr_acc  = max(onset_acc_min_ms2,  acc_mean  + onset_k_sd·acc_sd)
 *     C_on(i) := gyro_s > thr_gyro OR acc_s > thr_acc. Evaluated from the sample AFTER markGo.
 *  5. ARMED: when C_on holds continuously for onset_sustain_ms (run starts at j0) → onset is
 *     back-dated: let k_min = first index with t ≥ go_t − 300; search k = j0−1 down to k_min for the
 *     largest k with C_still(k) (using the stored smoothed values); onset_index = k + 1 if found, else
 *     k_min (never > j0). Emit {type:'onset', t: t_onset_index, index: onset_index} → MOVING.
 *     If t_i − go_t > max_wait_ms before an onset → emit 'no_movement' → FAILED.
 *  6. MOVING: peaks = max gyro_s / acc_s over samples from onset_index to the current sample.
 *     C_end(i) := gyro_s < end_gyro_dps AND acc_s < end_acc_ms2 AND t_i − t_onset ≥ min_movement_ms.
 *     When C_end holds continuously for end_sustain_ms (run starts at c0) → emit
 *     {type:'end', t: t_c0, index: c0, detected_t: t_i} → ENDED. Else if t_i − t_onset >
 *     max_movement_ms → emit 'too_long' → FAILED. (Checked in this order on each sample.)
 *  ENDED / FAILED: further samples are ignored (no events). result(): duration_ms = t_end − t_onset;
 *  peaks/indices null when not reached; baseline null before GO.
 *
 * LITERAL READINGS of points the text leaves open (segment.py must make the same choices):
 *  a. Indices. The segmenter keeps a per-sample history in push order; an index in an event or in
 *     result() is the `index` field of the sample at that history position (the position itself if
 *     `index` is not a non-negative integer). The TrialController pushes every recorded motion
 *     sample, already rounded exactly as stored, so position == stream index and a re-run on the
 *     stored record (streams.motion columns, GO at timing.go_ms) reproduces the online result.
 *  b. Arithmetic. |v| = sqrt(x*x + y*y + z*z) (not hypot); an EMA step is S + k*(x − S) with
 *     k = 1 − exp(−dt/τ); baseline mean = (sequential sum)/n, sd = sqrt(Σ(x − mean)²/n).
 *  c. "Present" = all 3 components finite, for acc, accg and rr alike.
 *  d. 'highpass': a sample without accg does not update G (G_i = G_{i−1}); the next update uses that
 *     sample's own dt_i (to the immediately preceding sample), like every EMA here.
 *  e. A sample is processed by the state the segmenter is in when the sample arrives; after a
 *     transition the new state's checks and runs start with the NEXT sample (STILL after the 'still'
 *     sample, C_on after the GO sample, C_end after the sample that confirmed the onset). Hence at
 *     most one event per sample.
 *  f. ARMED: the onset check comes before the max_wait check on the same sample.
 *  g. Peaks cover onset_index through the last sample processed in MOVING (the end-confirmation or
 *     the too_long sample).
 *  h. stillProgress is 1 in every state after WAITING_STILL; with still_hold_ms ≤ 0 it is 1 whenever
 *     C_still holds.
 *  i. markGo() outside STILL, or with a non-finite t, is ignored (it never throws: it runs inside
 *     sensor handlers).
 *  j. Inputs that cannot occur in a recorded stream: a sample with non-finite t is ignored; a
 *     decreasing t is treated as dt = 0 (t must be non-decreasing; equal times give dt = 0, k = 0).
 *  k. acc_source is 'none' until the first sample; `live` is frozen once ENDED / FAILED.
 *  l. runOnlineSegmentation: see its own doc comment.
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import type { Baseline, MotionSample, SegmentationParams, SegmentationResult, Vec3 } from './types';

export type SegmenterState = 'waiting_still' | 'still' | 'armed' | 'moving' | 'ended' | 'failed';

export type SegmenterEvent =
  | { type: 'still'; t: number }
  | { type: 'still_lost'; t: number }
  | { type: 'false_start'; t: number }
  | { type: 'onset'; t: number; index: number }
  | { type: 'end'; t: number; index: number; detected_t: number }
  | { type: 'no_movement'; t: number }
  | { type: 'too_long'; t: number };

/** A motion sample whose `t` is already relative to the trial t0, plus its index in the stream. */
export interface RelMotionSample extends MotionSample {
  index: number;
}

export interface SegmenterLive {
  gyro: number;
  acc: number;
  stillProgress: number;
}

/** The onset is never back-dated earlier than this before GO (spec constant, not a parameter). */
export const ONSET_LOOKBACK_MS = 300;

type AccSource = SegmentationResult['acc_source'];

function isVec3(v: Vec3 | null | undefined): v is Vec3 {
  return !!v && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

function norm3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

/** One EMA step (spec 1). τ ≤ 0 means no smoothing. */
function emaStep(prev: number, x: number, dt: number, tau: number): number {
  if (!(tau > 0)) return x;
  const k = 1 - Math.exp(-dt / tau);
  return prev + k * (x - prev);
}

export class OnlineSegmenter {
  private readonly p: SegmentationParams;
  private st: SegmenterState = 'waiting_still';
  private accSource: AccSource | null = null;

  // Per-sample history in push order: back-dating, peaks and the baseline all look backwards.
  private readonly hT: number[] = [];
  private readonly hIndex: number[] = [];
  private readonly hGyro: number[] = []; // smoothed
  private readonly hAcc: number[] = []; // smoothed
  private readonly hStill: boolean[] = [];
  private readonly hAccg: (Vec3 | null)[] = []; // raw, for the baseline gravity

  private gyroRaw = 0;
  private accRaw = 0;
  private gyroS = 0;
  private accS = 0;
  private grav: Vec3 | null = null; // high-pass gravity estimate G
  private progress = 0;

  /** history position where the current run of the active state's condition began (−1: none) */
  private run = -1;
  /** first position of the still run that led to STILL = start of the baseline buffer */
  private bufStart = -1;
  private goT: number | null = null;
  private thrGyro = 0;
  private thrAcc = 0;
  private base: Baseline | null = null;
  private onsetPos = -1;
  private endPos = -1;
  private peakGyro = 0;
  private peakAcc = 0;

  constructor(params: SegmentationParams) {
    // Copy: the result must report the parameters that were actually used.
    this.p = { ...params };
  }

  /** Feed one sample (strictly increasing t). Returns the events this sample triggered (usually none). */
  push(s: RelMotionSample): SegmenterEvent[] {
    if (this.st === 'ended' || this.st === 'failed') return [];
    const t = s.t;
    if (typeof t !== 'number' || !Number.isFinite(t)) return [];
    const p = this.p;
    const pos = this.hT.length;
    const dt = pos === 0 ? 0 : Math.max(0, t - this.hT[pos - 1]);

    // 1. raw signals
    const rr = s.rr;
    if (rr && Number.isFinite(rr.alpha) && Number.isFinite(rr.beta) && Number.isFinite(rr.gamma)) {
      this.gyroRaw = norm3(rr.alpha, rr.beta, rr.gamma);
    }
    const accg = isVec3(s.accg) ? s.accg : null;
    if (pos === 0) this.accSource = isVec3(s.acc) ? 'linear' : accg ? 'highpass' : 'none';
    if (this.accSource === 'linear') {
      const a = s.acc;
      if (isVec3(a)) this.accRaw = norm3(a[0], a[1], a[2]);
    } else if (this.accSource === 'highpass' && accg) {
      const g = this.grav;
      if (g === null) {
        this.grav = [accg[0], accg[1], accg[2]];
      } else {
        g[0] = emaStep(g[0], accg[0], dt, p.gravity_tau_ms);
        g[1] = emaStep(g[1], accg[1], dt, p.gravity_tau_ms);
        g[2] = emaStep(g[2], accg[2], dt, p.gravity_tau_ms);
      }
      const G = this.grav as Vec3;
      this.accRaw = norm3(accg[0] - G[0], accg[1] - G[1], accg[2] - G[2]);
    }
    if (pos === 0) {
      this.gyroS = this.gyroRaw;
      this.accS = this.accRaw;
    } else {
      this.gyroS = emaStep(this.gyroS, this.gyroRaw, dt, p.smoothing_ms);
      this.accS = emaStep(this.accS, this.accRaw, dt, p.smoothing_ms);
    }
    const gyro = this.gyroS;
    const acc = this.accS;
    const still = gyro < p.still_gyro_dps && acc < p.still_acc_ms2;

    this.hT.push(t);
    this.hIndex.push(Number.isInteger(s.index) && s.index >= 0 ? s.index : pos);
    this.hGyro.push(gyro);
    this.hAcc.push(acc);
    this.hStill.push(still);
    this.hAccg.push(accg ? [accg[0], accg[1], accg[2]] : null);

    const events: SegmenterEvent[] = [];
    switch (this.st) {
      case 'waiting_still': {
        if (!still) {
          this.run = -1;
          this.progress = 0;
          break;
        }
        if (this.run < 0) this.run = pos;
        const held = t - this.hT[this.run];
        this.progress = p.still_hold_ms > 0 ? Math.min(1, held / p.still_hold_ms) : 1;
        if (held >= p.still_hold_ms) {
          this.bufStart = this.run;
          this.run = -1;
          this.progress = 1;
          this.st = 'still';
          events.push({ type: 'still', t });
        }
        break;
      }
      case 'still': {
        // A C_still sample simply extends the buffer (it is the contiguous range bufStart..pos).
        if (still) break;
        const early = gyro > p.onset_gyro_min_dps || acc > p.onset_acc_min_ms2;
        events.push({ type: early ? 'false_start' : 'still_lost', t });
        this.st = 'waiting_still';
        this.bufStart = -1;
        this.run = -1;
        this.progress = 0;
        break;
      }
      case 'armed': {
        if (gyro > this.thrGyro || acc > this.thrAcc) {
          if (this.run < 0) this.run = pos;
          if (t - this.hT[this.run] >= p.onset_sustain_ms) {
            this.confirmOnset(this.run, pos, events);
            break; // reading f: an onset on this sample pre-empts no_movement
          }
        } else {
          this.run = -1;
        }
        if (t - (this.goT as number) > p.max_wait_ms) {
          events.push({ type: 'no_movement', t });
          this.st = 'failed';
        }
        break;
      }
      case 'moving': {
        if (gyro > this.peakGyro) this.peakGyro = gyro;
        if (acc > this.peakAcc) this.peakAcc = acc;
        const tOnset = this.hT[this.onsetPos];
        if (gyro < p.end_gyro_dps && acc < p.end_acc_ms2 && t - tOnset >= p.min_movement_ms) {
          if (this.run < 0) this.run = pos;
          if (t - this.hT[this.run] >= p.end_sustain_ms) {
            this.endPos = this.run;
            this.run = -1;
            this.st = 'ended';
            events.push({ type: 'end', t: this.hT[this.endPos], index: this.hIndex[this.endPos], detected_t: t });
            break;
          }
        } else {
          this.run = -1;
        }
        if (t - tOnset > p.max_movement_ms) {
          events.push({ type: 'too_long', t });
          this.st = 'failed';
        }
        break;
      }
      default:
        break;
    }
    return events;
  }

  /** Spec 5: back-dates the onset of the C_on run that started at j0 and was confirmed at `pos`. */
  private confirmOnset(j0: number, pos: number, events: SegmenterEvent[]): void {
    const lookFrom = (this.goT as number) - ONSET_LOOKBACK_MS;
    let kMin = j0;
    for (let k = 0; k <= j0; k++) {
      if (this.hT[k] >= lookFrom) {
        kMin = k;
        break;
      }
    }
    let onset = kMin;
    for (let k = j0 - 1; k >= kMin; k--) {
      if (this.hStill[k]) {
        onset = k + 1;
        break;
      }
    }
    if (onset > j0) onset = j0;
    this.onsetPos = onset;
    let pg = this.hGyro[onset];
    let pa = this.hAcc[onset];
    for (let k = onset + 1; k <= pos; k++) {
      if (this.hGyro[k] > pg) pg = this.hGyro[k];
      if (this.hAcc[k] > pa) pa = this.hAcc[k];
    }
    this.peakGyro = pg;
    this.peakAcc = pa;
    this.run = -1;
    this.st = 'moving';
    events.push({ type: 'onset', t: this.hT[onset], index: this.hIndex[onset] });
  }

  /** The controller emitted the GO cue at time t (relative ms). Only valid in state 'still'. */
  markGo(t: number): void {
    if (this.st !== 'still' || typeof t !== 'number' || !Number.isFinite(t)) return;
    const p = this.p;
    const from = this.bufStart;
    const to = this.hT.length - 1;
    const n = to - from + 1;
    let sumG = 0;
    let sumA = 0;
    for (let k = from; k <= to; k++) {
      sumG += this.hGyro[k];
      sumA += this.hAcc[k];
    }
    const gyroMean = sumG / n;
    const accMean = sumA / n;
    let varG = 0;
    let varA = 0;
    let gx = 0;
    let gy = 0;
    let gz = 0;
    let ng = 0;
    for (let k = from; k <= to; k++) {
      const dg = this.hGyro[k] - gyroMean;
      varG += dg * dg;
      const da = this.hAcc[k] - accMean;
      varA += da * da;
      const g = this.hAccg[k];
      if (g) {
        gx += g[0];
        gy += g[1];
        gz += g[2];
        ng++;
      }
    }
    const gyroSd = Math.sqrt(varG / n);
    const accSd = Math.sqrt(varA / n);
    this.base = {
      n,
      gyro_mean: gyroMean,
      gyro_sd: gyroSd,
      acc_mean: accMean,
      acc_sd: accSd,
      gravity: ng > 0 ? [gx / ng, gy / ng, gz / ng] : null,
    };
    this.thrGyro = Math.max(p.onset_gyro_min_dps, gyroMean + p.onset_k_sd * gyroSd);
    this.thrAcc = Math.max(p.onset_acc_min_ms2, accMean + p.onset_k_sd * accSd);
    this.goT = t;
    this.run = -1;
    this.st = 'armed';
  }

  get state(): SegmenterState {
    return this.st;
  }

  get live(): SegmenterLive {
    return { gyro: this.gyroS, acc: this.accS, stillProgress: this.progress };
  }

  /** Baseline frozen at GO (null before GO). */
  get baseline(): Baseline | null {
    const b = this.base;
    if (!b) return null;
    return { ...b, gravity: b.gravity ? [b.gravity[0], b.gravity[1], b.gravity[2]] : null };
  }

  /** Onset thresholds frozen at GO (null before GO). For diagnostics and tests. */
  get thresholds(): { gyro: number; acc: number } | null {
    return this.base ? { gyro: this.thrGyro, acc: this.thrAcc } : null;
  }

  /** The t passed to markGo (null before GO). */
  get goTime(): number | null {
    return this.goT;
  }

  result(): SegmentationResult {
    const onset = this.onsetPos >= 0;
    const end = this.endPos >= 0;
    return {
      method: 'online_v1',
      params: { ...this.p },
      acc_source: this.accSource ?? 'none',
      onset_index: onset ? this.hIndex[this.onsetPos] : null,
      end_index: end ? this.hIndex[this.endPos] : null,
      duration_ms: onset && end ? this.hT[this.endPos] - this.hT[this.onsetPos] : null,
      peak_gyro_dps: onset ? this.peakGyro : null,
      peak_acc_ms2: onset ? this.peakAcc : null,
      baseline: this.baseline,
    };
  }
}

/**
 * The smoothed activity signals of spec step 1 (gyro_s = EMA |ω|, acc_s = EMA |a_lin| with the same
 * acc_source rule), WITHOUT any state machine: identical arithmetic to OnlineSegmenter (a test pins
 * that both give the same values sample by sample), but it keeps running after the segmenter has
 * ENDED / FAILED, whose `live` is frozen. The TrialController uses it for what happens around the
 * segmented movement (the return of the arm, the false-start policy). Not part of online_v1: the
 * Python port does not need it.
 */
export class MotionSignals {
  private readonly smoothingMs: number;
  private readonly gravityTauMs: number;
  private accSource: AccSource | null = null;
  private lastT: number | null = null;
  private gyroRaw = 0;
  private accRaw = 0;
  private gyroS = 0;
  private accS = 0;
  private grav: Vec3 | null = null;

  constructor(params: Pick<SegmentationParams, 'smoothing_ms' | 'gravity_tau_ms'>) {
    this.smoothingMs = params.smoothing_ms;
    this.gravityTauMs = params.gravity_tau_ms;
  }

  /** Feeds one sample (non-decreasing t); returns the smoothed values after it (unchanged for a non-finite t). */
  push(s: MotionSample): { gyro: number; acc: number } {
    const t = s.t;
    if (typeof t !== 'number' || !Number.isFinite(t)) return { gyro: this.gyroS, acc: this.accS };
    const first = this.lastT === null;
    const dt = first ? 0 : Math.max(0, t - (this.lastT as number));
    this.lastT = t;
    const rr = s.rr;
    if (rr && Number.isFinite(rr.alpha) && Number.isFinite(rr.beta) && Number.isFinite(rr.gamma)) {
      this.gyroRaw = norm3(rr.alpha, rr.beta, rr.gamma);
    }
    const accg = isVec3(s.accg) ? s.accg : null;
    if (first) this.accSource = isVec3(s.acc) ? 'linear' : accg ? 'highpass' : 'none';
    if (this.accSource === 'linear') {
      const a = s.acc;
      if (isVec3(a)) this.accRaw = norm3(a[0], a[1], a[2]);
    } else if (this.accSource === 'highpass' && accg) {
      const g = this.grav;
      if (g === null) {
        this.grav = [accg[0], accg[1], accg[2]];
      } else {
        g[0] = emaStep(g[0], accg[0], dt, this.gravityTauMs);
        g[1] = emaStep(g[1], accg[1], dt, this.gravityTauMs);
        g[2] = emaStep(g[2], accg[2], dt, this.gravityTauMs);
      }
      const G = this.grav as Vec3;
      this.accRaw = norm3(accg[0] - G[0], accg[1] - G[1], accg[2] - G[2]);
    }
    if (first) {
      this.gyroS = this.gyroRaw;
      this.accS = this.accRaw;
    } else {
      this.gyroS = emaStep(this.gyroS, this.gyroRaw, dt, this.smoothingMs);
      this.accS = emaStep(this.accS, this.accRaw, dt, this.smoothingMs);
    }
    return { gyro: this.gyroS, acc: this.accS };
  }

  get gyro(): number {
    return this.gyroS;
  }

  get acc(): number {
    return this.accS;
  }
}

/**
 * Convenience for tests/offline re-runs: feeds all samples, calling markGo at the first sample with
 * t ≥ goT (if goT is null, GO is given automatically `foreperiodMs` after the 'still' event).
 *
 * Literal readings (reading l; analysis/motionlab/segment.py run_online_v1 makes the same choices):
 *  - GO is always given AT a sample, as markGo(t of that sample) — exactly what the TrialController
 *    does — so the result does not depend on where goT falls between two samples.
 *  - goT given: GO right after pushing the first sample with t ≥ goT. It is attempted once only: if
 *    the segmenter is not in 'still' at that sample, no GO is given. To reproduce a recorded trial
 *    pass goT = timing.go_ms, which is exactly the t of the sample GO was given at.
 *  - goT null: like the TrialController — each 'still' event schedules GO at still_t + foreperiodMs
 *    (default params.foreperiod_min_ms); 'still_lost' / 'false_start' cancel it. GO is given at the
 *    first sample (possibly the 'still' sample itself when foreperiodMs is 0) with t ≥ the scheduled
 *    time while in 'still'.
 *  - The returned goT is the time passed to markGo (a sample time), or null when GO was never given.
 */
export function runOnlineSegmentation(
  samples: RelMotionSample[],
  params: SegmentationParams,
  goT: number | null,
  foreperiodMs?: number,
): { events: SegmenterEvent[]; result: SegmentationResult; goT: number | null } {
  const seg = new OnlineSegmenter(params);
  const events: SegmenterEvent[] = [];
  const foreperiod = foreperiodMs ?? params.foreperiod_min_ms;
  let usedGo: number | null = null;
  let goDone = false;
  let pendingAt: number | null = null;
  for (const s of samples) {
    for (const e of seg.push(s)) {
      events.push(e);
      if (e.type === 'still') pendingAt = e.t + foreperiod;
      else if (e.type === 'still_lost' || e.type === 'false_start') pendingAt = null;
    }
    if (goDone) continue;
    if (goT !== null) {
      if (s.t >= goT) {
        goDone = true;
        if (seg.state === 'still') {
          seg.markGo(s.t);
          usedGo = s.t;
        }
      }
    } else if (pendingAt !== null && seg.state === 'still' && s.t >= pendingAt) {
      goDone = true;
      seg.markGo(s.t);
      usedGo = s.t;
    }
  }
  return { events, result: seg.result(), goT: usedGo };
}
