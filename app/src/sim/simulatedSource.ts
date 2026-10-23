/**
 * A SensorSource that plays a simulated participant performing the instructed trial.
 *
 * Behaviour timeline (source clock):
 *  - from start: holding the phone at p0 (extended arm) with physiological tremor (≈ 7–12 Hz,
 *    style.tremor_dps) plus slow postural sway, and sensor white noise; still enough to satisfy the
 *    default hold-still criterion (smoothed |ω| ≈ 2–4 deg/s, |a_lin| ≈ 0.05–0.15 m/s²).
 *  - notifyCue('go', t): movement onset at t + reaction (style.reaction_ms ± 25%).
 *    behavior 'false_start': once the participant has been still for 800 ms (the default
 *    still_hold_ms 500 + 300 ms, i.e. inside the foreperiod, before the earliest possible GO), it
 *    makes a small abrupt anticipatory jerk (12 mm forward + 6° pitch, 250 ms, acceleration step
 *    ≈ 3.8 m/s² at its start so the segmenter classifies it as a false start rather than
 *    'still_lost'), then returns to stillness; a GO arriving before that cancels it.
 *    behavior 'no_move': ignores GO. behavior 'too_slow': the movement is stretched to 20 s and a
 *    continuous circular wobble (≈ 45 deg/s) is superimposed — a slow drift alone would never
 *    cross the onset thresholds, the wobble guarantees onset AND no end (→ 'too_long').
 *  - movement per planMovement; at s = 1 a short contact transient (60 ms biphasic pulse along the
 *    body's forward axis — device ∓z for screen toward/away — ≤ 3 m/s² for fast movements); then
 *    holds at the nose with tremor.
 *  - notifyCue('end', t): after 250–450 ms (and never before the phone has arrived) starts
 *    returning the arm (reverse minimum jerk along the same path, 0.8 s).
 *  - mode 'realtime': setInterval at the platform rate, t = performance.now(); 'offline': start()
 *    synchronously loops virtual time (from opts.startTime, default 1000 ms) until stop() is called
 *    or opts.maxDurationMs (default 60000) elapses, then resolves. In realtime mode maxDurationMs
 *    only applies when given explicitly.
 *  - ground truth is exposed for tests: truth.go_t, truth.onset_t (movement begins), truth.end_t
 *    (phone reaches the nose), in the source clock.
 *
 * Platform realism: gravitySign multiplies acc and accg; rotationRate uses the W3C 2025 field
 * mapping (alpha/beta/gamma = device x/y/z) unless platform.rr_mapping is 'legacy' (alpha = z,
 * beta = x, gamma = y); acc_available false → acc null;
 * quaternion true → onQuaternion events (RelativeOrientationSensor, own 60 Hz schedule);
 * deviceorientation events right after every motion event (absolute: false, compass null — not
 * modelled); ts = t − 0.3…2 ms dispatch latency, and sensor values are evaluated at ts; handler
 * times jitter by N(0, jitter_ms) (clamped to ±40% of the period); single drops and rare 3–10
 * sample stalls with total probability ≈ drop_prob. event.interval: Android/Chrome reports 16 (ms),
 * iOS/WebKit reports CoreMotion's update interval in SECONDS (float32 1/60 ≈ 0.0166667).
 * platform.quantize → every emitted value is rounded on Chromium's grids (CHROMIUM_ROUNDING, same
 * arithmetic as RoundToMultiple), after noise and bias, like a reading of Chrome on Android.
 * Chromium also skips deviceorientation events whose angles moved < 0.1°; NOT modelled (one
 * orientation event per motion event either way).
 * opts.noise === false switches off every stochastic part except the movement plan itself (sensor
 * noise, gyro bias, tremor/sway, timing jitter/latency spread, drops) — for exact tests — and with
 * it the quantisation (a measurement artefact, not part of the movement).
 * Random streams (plan, timing, tremor, sensor noise, quaternion timing) are seeded separately
 * from opts.seed, so turning noise off never changes the movement draws.
 *
 * IMPLEMENTED BY: simulation agent.
 */
import {
  DEG,
  eulerZXYFromMat,
  mExp,
  mLog,
  mMul,
  mT,
  mVec,
  quatFromMat,
  vectorToRotationRate,
  type Mat3,
} from '../core/math3d';
import type {
  CueKind,
  MotionSample,
  OrientationSample,
  QuaternionSample,
  SensorListener,
  SensorSource,
  SourceClock,
  TrialCondition,
  Vec3,
} from '../core/types';
import {
  BODY_GRAVITY,
  deriveSeed,
  earthFromBody,
  gaussian,
  gripMatrix,
  minJerk,
  planMovement,
  simRng,
  type MovementPlan,
  type SimParticipant,
  type SimPlatform,
} from './model';

/**
 * normal: performs the instructed movement after GO. false_start: one anticipatory movement before GO.
 * no_move: holds still forever (with tremor). too_slow: movement lasting 20 s.
 * static: phone lying motionless (no tremor, tiny sensor noise only) in the grip orientation of the
 *         condition, e.g. flat_up for the "phone on a table" calibration.
 * tilt: the calibration tilt gesture — held (with tremor) in the condition's nominal grip, 1 s after
 *       start tilts 45° about the device y axis (right edge down; 0.8 s), holds 0.6 s, tilts back
 *       (0.8 s). Ignores cues. truth.onset_t / end_t = start / end of the gesture.
 */
export type SimBehavior = 'normal' | 'false_start' | 'no_move' | 'too_slow' | 'static' | 'tilt';

export interface SimSourceOptions {
  mode: 'realtime' | 'offline';
  participant: SimParticipant;
  platform: SimPlatform;
  condition: TrialCondition;
  seed: number;
  behavior?: SimBehavior;
  startTime?: number; // offline only
  baseEpochMs?: number; // epochAt(t) = baseEpochMs + t (offline); realtime uses performance.timeOrigin
  maxDurationMs?: number;
  /** false → noise-free signals (no sensor noise/bias, tremor, timing jitter or drops). Default true. */
  noise?: boolean;
}

export interface SimTruth {
  go_t: number | null;
  onset_t: number | null;
  end_t: number | null;
  behavior: SimBehavior;
  /** start of the anticipatory jerk (behavior 'false_start'), null if none happened/planned */
  false_start_t?: number | null;
  /** start of the return movement after the 'end' cue */
  return_t?: number | null;
}

export interface SimulatedSource extends SensorSource {
  readonly truth: SimTruth;
  /** the movement this source performs (drawn at creation) */
  readonly plan: MovementPlan;
}

/** Time the participant must have been still before the false-start jerk (ms). */
export const FALSE_START_AFTER_STILL_MS = 800;
const FALSE_START_MS = 250;
const FALSE_START_PUSH_M = 0.012;
const FALSE_START_PITCH_DEG = 6;
const CONTACT_MS = 60;
const RETURN_MS = 800;
const TOO_SLOW_MS = 20000;
const WOBBLE_DEG = 6;
const WOBBLE_HZ = 1.2;
const WOBBLE_RAMP_S = 0.25;
/** tilt gesture: start after start(), angle, and the durations of its three phases (ms) */
export const TILT_START_MS = 1000;
export const TILT_DEG = 45;
const TILT_MOVE_MS = 800;
const TILT_HOLD_MS = 600;
/** half-step of the central difference used for ω (s) */
const FD_S = 0.0005;

/**
 * Chromium's privacy rounding grids (services/device/generic_sensor/platform_sensor_util.h; applied
 * to every reading in PlatformSensor::UpdateSharedBuffer, so devicemotion, deviceorientation and
 * the Generic Sensor API all see rounded values). WebKit does not round.
 */
export const CHROMIUM_ROUNDING = {
  acc_ms2: 0.1, // accelerometer, linear acceleration, gravity
  gyro_rad_s: 0.00174532925199432963, // 0.1 deg/s; Blink converts to deg/s AFTER rounding
  euler_deg: 0.1,
  quat_rad: 0.0017453292519943296, // 0.1° for the rotation angle and the axis azimuth/elevation
} as const;

/** Chromium RoundToMultiple: the nearest multiple of `multiple`, halves away from zero. */
export function roundToMultiple(value: number, multiple: number): number {
  const scaled = value / multiple;
  return value < 0 ? -multiple * Math.floor(-scaled + 0.5) : multiple * Math.floor(scaled + 0.5);
}

/** Chromium RoundOrientationQuaternionReading (unit quaternion [x, y, z, w]); stays unit length. */
export function roundQuaternionLikeChromium(q: [number, number, number, number]): [number, number, number, number] {
  const step = CHROMIUM_ROUNDING.quat_rad;
  const half = Math.acos(Math.max(-1, Math.min(1, q[3])));
  const roundedHalf = roundToMultiple(half * 2, step) / 2;
  if (roundedHalf === 0) return [0, 0, 0, 1];
  const s = Math.sin(half);
  const ax = q[0] / s;
  const ay = q[1] / s;
  const az = q[2] / s;
  const azim = roundToMultiple(Math.atan2(ax, ay), step);
  const elev = roundToMultiple(Math.atan2(az, Math.sqrt(ax * ax + ay * ay)), step);
  const sr = Math.sin(roundedHalf);
  return [Math.sin(azim) * Math.cos(elev) * sr, Math.cos(azim) * Math.cos(elev) * sr, Math.sin(elev) * sr, Math.cos(roundedHalf)];
}

function roundAcc(v: Vec3): Vec3 {
  const m = CHROMIUM_ROUNDING.acc_ms2;
  return [roundToMultiple(v[0], m), roundToMultiple(v[1], m), roundToMultiple(v[2], m)];
}

/** deg/s → rad/s → Chromium grid → deg/s (base::RadToDeg), so the float noise matches Chrome's. */
function roundGyroDeg(v: Vec3): Vec3 {
  const m = CHROMIUM_ROUNDING.gyro_rad_s;
  const r = (x: number) => roundToMultiple(x * (Math.PI / 180), m) * (180 / Math.PI);
  return [r(v[0]), r(v[1]), r(v[2])];
}

interface Osc {
  amp: number; // rad (rotation) or m (position)
  w: number; // rad/s
  ph: number; // rad
}

interface Tremor {
  rot: Osc[][]; // per body axis
  pos: Osc[][];
}

function drawOscs(rng: () => number, n: number, fLo: number, fHi: number, ampOf: (w: number) => number): Osc[] {
  const out: Osc[] = [];
  for (let i = 0; i < n; i++) {
    const w = 2 * Math.PI * (fLo + (fHi - fLo) * rng());
    const k = rng();
    const ph = 2 * Math.PI * rng();
    out.push({ amp: ampOf(w) * (0.6 + 0.8 * k), w, ph });
  }
  return out;
}

/**
 * Physiological tremor (7–12 Hz) + postural sway (0.15–0.6 Hz). The rotation part has an angular-
 * velocity RMS ≈ tremorDps; the translation part an acceleration RMS ≈ 0.03·jf m/s².
 */
function buildTremor(rng: () => number, tremorDps: number, jf: number): Tremor {
  const perOsc = tremorDps / Math.sqrt(3); // 2 oscillators per axis → per-axis RMS = amplitude
  const accPerOsc = (0.03 * jf) / Math.sqrt(3);
  const rot: Osc[][] = [];
  const pos: Osc[][] = [];
  for (let j = 0; j < 3; j++) {
    rot.push([
      ...drawOscs(rng, 2, 7, 12, (w) => (perOsc * DEG) / w),
      ...drawOscs(rng, 1, 0.15, 0.6, () => 0.25 * jf * DEG),
    ]);
    pos.push([
      ...drawOscs(rng, 2, 7, 12, (w) => accPerOsc / (w * w)),
      ...drawOscs(rng, 1, 0.15, 0.6, () => 0.0015 * jf),
    ]);
  }
  return { rot, pos };
}

export function createSimulatedSource(opts: SimSourceOptions): SimulatedSource {
  const behavior: SimBehavior = opts.behavior ?? 'normal';
  const noise = opts.noise !== false;
  const P = opts.participant;
  const pf = opts.platform;
  const mode = opts.mode;
  const maxDurationMs = opts.maxDurationMs ?? (mode === 'offline' ? 60000 : Infinity);
  const periodMs = 1000 / pf.hz;

  // independent random streams
  const planRng = simRng(deriveSeed(opts.seed, 0));
  const timeRng = simRng(deriveSeed(opts.seed, 1));
  const tremorRng = simRng(deriveSeed(opts.seed, 2));
  const noiseRng = simRng(deriveSeed(opts.seed, 3));
  const quatRng = simRng(deriveSeed(opts.seed, 4));

  const plan = planMovement(P, opts.condition, planRng);
  const reactionMs = P.style.reaction_ms * (1 + 0.25 * (2 * planRng() - 1));
  const returnDelayMs = 250 + 200 * planRng();
  const h = plan.h;
  const jf = plan.nondominant ? P.style.nondominant_jitter : 1;
  const moveMs = behavior === 'too_slow' ? TOO_SLOW_MS : plan.duration_s * 1000;
  const tremor = noise && behavior !== 'static' ? buildTremor(tremorRng, P.style.tremor_dps * jf, jf) : null;
  const staticR = gripMatrix(plan.orientation); // a phone lying on a table has no grip tilt
  const R_EB = earthFromBody(pf.heading_deg);
  const accNoise = noise ? pf.acc_noise_ms2 : 0;
  const gyroNoise = noise ? pf.gyro_noise_dps : 0;
  const bias: Vec3 = noise ? pf.gyro_bias_dps : [0, 0, 0];
  const quantize = noise && pf.quantize === true;
  const interval = pf.name === 'ios' ? Math.fround(1 / pf.hz) : Math.floor(1000 / pf.hz);

  const truth: SimTruth = {
    go_t: null,
    onset_t: null,
    end_t: null,
    behavior,
    false_start_t: null,
    return_t: null,
  };

  let startT = opts.startTime ?? 1000;
  let vt = startT; // offline virtual clock
  let started = false;
  let stopped = false;
  let timer: ReturnType<typeof setInterval> | null = null;
  let onsetT: number | null = null;
  let arriveT: number | null = null;
  let returnT: number | null = null;
  let fsT: number | null = null;

  // offline default epoch: the start of the virtual timeline corresponds to "now"
  const baseEpochMs = opts.baseEpochMs ?? Date.now() - startT;
  const clock: SourceClock =
    mode === 'offline'
      ? { now: () => vt, epochAt: (t) => baseEpochMs + t }
      : { now: () => performance.now(), epochAt: (t) => performance.timeOrigin + t };

  // ── kinematics ──

  function basePose(t: number): { R: Mat3; a: Vec3 } {
    if (behavior === 'static') return { R: staticR, a: [0, 0, 0] };
    if (behavior === 'tilt') {
      // rotation about the DEVICE y axis → right-multiply the grip
      const tau = t - (startT + TILT_START_MS);
      const up = minJerk(tau / TILT_MOVE_MS).u;
      const down = minJerk((tau - TILT_MOVE_MS - TILT_HOLD_MS) / TILT_MOVE_MS).u;
      const ang = TILT_DEG * DEG * (up - down);
      return { R: ang === 0 ? staticR : mMul(staticR, mExp([0, ang, 0])), a: [0, 0, 0] };
    }
    if (onsetT === null || t <= onsetT) return { R: plan.grip, a: [0, 0, 0] };
    if (returnT !== null && t > returnT) {
      const m = minJerk((t - returnT) / RETURN_MS);
      const d = RETURN_MS / 1000;
      const st = plan.path(1 - m.u, -m.du / d, -m.ddu / (d * d));
      return { R: st.R, a: st.a };
    }
    const ts = (t - onsetT) / 1000;
    if (behavior === 'too_slow') {
      const k = plan.duration_s / (TOO_SLOW_MS / 1000); // time-stretch of the planned movement
      const st = plan.pose(ts * k);
      return { R: st.R, a: [st.a[0] * k * k, st.a[1] * k * k, st.a[2] * k * k] };
    }
    const st = plan.pose(ts);
    return { R: st.R, a: st.a };
  }

  /** Full body-frame pose: base movement + tremor + behaviour-specific perturbations. */
  function poseAt(t: number): { R: Mat3; a: Vec3 } {
    const base = basePose(t);
    const a: Vec3 = [base.a[0], base.a[1], base.a[2]];
    const rv: Vec3 = [0, 0, 0]; // body-frame rotation vector applied on the left of R
    let rotate = false;
    const tsec = t / 1000;

    if (tremor) {
      // rotation: axial vector → y, z flip with the hand; translation: polar → x flips
      for (let j = 0; j < 3; j++) {
        const sr = j === 0 ? 1 : h;
        const sp = j === 0 ? h : 1;
        for (const o of tremor.rot[j]) rv[j] += sr * o.amp * Math.sin(o.w * tsec + o.ph);
        for (const o of tremor.pos[j]) a[j] -= sp * o.amp * o.w * o.w * Math.sin(o.w * tsec + o.ph);
      }
      rotate = true;
    }

    if (fsT !== null && t > fsT && t < fsT + FALSE_START_MS) {
      // abrupt twitch: x(τ) = D·sin²(πτ/T) forward, pitch θ(τ) = A·sin²(πτ/T)
      const T = FALSE_START_MS / 1000;
      const tau = (t - fsT) / 1000;
      const s2 = Math.sin((Math.PI * tau) / T) ** 2;
      a[2] -= ((FALSE_START_PUSH_M * 2 * Math.PI * Math.PI) / (T * T)) * Math.cos((2 * Math.PI * tau) / T);
      rv[0] += FALSE_START_PITCH_DEG * DEG * s2;
      rotate = true;
    }

    if (behavior === 'too_slow' && onsetT !== null && t > onsetT) {
      const tw = (t - onsetT) / 1000;
      const env = minJerk(tw / WOBBLE_RAMP_S).u * minJerk((TOO_SLOW_MS / 1000 - tw) / WOBBLE_RAMP_S).u;
      const ph = 2 * Math.PI * WOBBLE_HZ * tw;
      rv[0] += WOBBLE_DEG * DEG * env * Math.sin(ph);
      rv[1] += h * WOBBLE_DEG * DEG * env * (1 - Math.cos(ph));
      rotate = true;
    }

    if (arriveT !== null && behavior !== 'too_slow' && t >= arriveT && t < arriveT + CONTACT_MS) {
      // the nose pushes the phone forward (−z body), biphasic so the net velocity change is zero
      a[2] -= plan.contact_ms2 * Math.sin((2 * Math.PI * (t - arriveT)) / CONTACT_MS);
    }

    return { R: rotate ? mMul(mExp(rv), base.R) : base.R, a };
  }

  // ── sample generation ──

  function gauss3(sd: number): Vec3 {
    if (sd === 0) return [0, 0, 0];
    return [gaussian(noiseRng) * sd, gaussian(noiseRng) * sd, gaussian(noiseRng) * sd];
  }

  function latency(rng: () => number): number {
    return noise ? 0.3 + 1.7 * rng() : 1;
  }

  function motionSample(t: number, ts: number): MotionSample {
    const { R, a } = poseAt(ts);
    const Rp = poseAt(ts + FD_S * 1000).R;
    const Rm = poseAt(ts - FD_S * 1000).R;
    const wB = mLog(mMul(Rp, mT(Rm)));
    const Rt = mT(R);
    const wD = mVec(Rt, wB);
    const k = 1 / (2 * FD_S * DEG); // rad per 2h → deg/s
    const linD = mVec(Rt, a);
    const fD = mVec(Rt, [a[0] + BODY_GRAVITY[0], a[1] + BODY_GRAVITY[1], a[2] + BODY_GRAVITY[2]]);
    const gs = pf.gravitySign;
    const nA = gauss3(accNoise);
    const nL = gauss3(accNoise * 0.5); // extra noise of the fused linear acceleration
    const nG = gauss3(gyroNoise);
    let accg: Vec3 = [gs * fD[0] + nA[0], gs * fD[1] + nA[1], gs * fD[2] + nA[2]];
    let acc: Vec3 | null = pf.acc_available
      ? [gs * linD[0] + nA[0] + nL[0], gs * linD[1] + nA[1] + nL[1], gs * linD[2] + nA[2] + nL[2]]
      : null;
    let w: Vec3 = [wD[0] * k + nG[0] + bias[0], wD[1] * k + nG[1] + bias[1], wD[2] * k + nG[2] + bias[2]];
    if (quantize) {
      // per component, so it commutes with the sign flip and the legacy field permutation
      accg = roundAcc(accg);
      if (acc) acc = roundAcc(acc);
      w = roundGyroDeg(w);
    }
    const rr = pf.rr_mapping === 'legacy' ? { alpha: w[2], beta: w[0], gamma: w[1] } : vectorToRotationRate(w);
    return { t, ts, acc, accg, rr, interval };
  }

  function earthPose(ts: number): Mat3 {
    const RE = mMul(R_EB, poseAt(ts).R);
    // fusion noise as a tiny earth-frame rotation keeps the angles inside the W3C ranges
    return noise ? mMul(mExp(gauss3(0.05 * DEG)), RE) : RE;
  }

  function orientationSample(t: number, ts: number): OrientationSample {
    const e = eulerZXYFromMat(earthPose(ts));
    if (quantize) {
      // like Chromium: no re-wrapping, so alpha can read 360.0 (beta 180.0, gamma 90.0) at the edges
      const m = CHROMIUM_ROUNDING.euler_deg;
      e.alpha = roundToMultiple(e.alpha, m);
      e.beta = roundToMultiple(e.beta, m);
      e.gamma = roundToMultiple(e.gamma, m);
    }
    return { t, ts, alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: false, compass: null };
  }

  function quaternionSample(t: number): QuaternionSample {
    const q = quatFromMat(earthPose(t));
    return { t, q: quantize ? roundQuaternionLikeChromium(q) : q, sensor: 'RelativeOrientationSensor' };
  }

  // ── timing (drops/stalls use a fixed number of draws per tick to keep streams aligned) ──

  function makeDropper(rng: () => number): () => boolean {
    let stallLeft = 0;
    return () => {
      const u = rng();
      const len = 3 + Math.floor(8 * rng());
      if (!noise) return false;
      if (stallLeft > 0) {
        stallLeft--;
        return true;
      }
      if (u < pf.drop_prob * 0.1) {
        stallLeft = len - 1;
        return true;
      }
      return u < pf.drop_prob;
    };
  }

  function jitter(rng: () => number): number {
    const z = gaussian(rng);
    return noise ? Math.max(-0.4 * periodMs, Math.min(0.4 * periodMs, z * pf.jitter_ms)) : 0;
  }

  // ── running ──

  function runOffline(listener: SensorListener): void {
    const limit = startT + maxDurationMs;
    const motionDrop = makeDropper(timeRng);
    const quatDrop = makeDropper(quatRng);
    const phase = noise ? periodMs * (0.2 + 0.6 * timeRng()) : periodMs / 2;
    const qPhase = noise ? periodMs * quatRng() : periodMs / 4;
    const wantQuat = pf.quaternion && typeof listener.onQuaternion === 'function';

    interface Tick {
      k: number;
      tm: number;
      to: number;
      dropped: boolean;
      latM: number;
      latO: number;
    }
    const makeTick = (k: number): Tick => {
      const tm = startT + phase + k * periodMs + jitter(timeRng);
      // orientation right after the motion event; < 0.2·period so it never passes the next tick
      const to = tm + (noise ? 0.2 + 1.3 * timeRng() : 0.5);
      return { k, tm, to, dropped: motionDrop(), latM: latency(timeRng), latO: latency(timeRng) };
    };
    const makeQ = (j: number): { j: number; t: number; dropped: boolean } => ({
      j,
      t: startT + qPhase + j * periodMs + jitter(quatRng),
      dropped: quatDrop(),
    });

    let tick = makeTick(0);
    let stage: 'motion' | 'orientation' = 'motion';
    let q = wantQuat ? makeQ(0) : null;
    while (!stopped) {
      const tTick = stage === 'motion' ? tick.tm : tick.to;
      if (q && q.t <= tTick) {
        if (q.t > limit) break;
        vt = q.t;
        if (!q.dropped) listener.onQuaternion?.(quaternionSample(q.t));
        q = makeQ(q.j + 1);
        continue;
      }
      if (tTick > limit) break;
      vt = tTick;
      if (stage === 'motion') {
        if (!tick.dropped) listener.onMotion(motionSample(tick.tm, tick.tm - tick.latM));
        stage = 'orientation';
      } else {
        if (!tick.dropped) listener.onOrientation(orientationSample(tick.to, tick.to - tick.latO));
        tick = makeTick(tick.k + 1);
        stage = 'motion';
      }
    }
    stopped = true;
  }

  function runRealtime(listener: SensorListener): void {
    const drop = makeDropper(timeRng);
    const wantQuat = pf.quaternion && typeof listener.onQuaternion === 'function';
    timer = setInterval(() => {
      if (stopped) return;
      const t = performance.now();
      if (t - startT > maxDurationMs) {
        source.stop();
        return;
      }
      if (drop()) return;
      listener.onMotion(motionSample(t, t - latency(timeRng)));
      if (stopped) return;
      const to = performance.now();
      listener.onOrientation(orientationSample(to, to - latency(timeRng)));
      if (stopped || !wantQuat) return;
      listener.onQuaternion?.(quaternionSample(performance.now()));
    }, periodMs);
  }

  const source: SimulatedSource = {
    kind: 'simulated',
    clock,
    truth,
    plan,
    start(listener: SensorListener): Promise<void> {
      if (started) return Promise.reject(new Error('simulated source: start() called twice'));
      started = true;
      if (stopped) return Promise.resolve();
      if (mode === 'realtime') startT = performance.now();
      vt = startT;
      if (behavior === 'false_start') {
        fsT = startT + FALSE_START_AFTER_STILL_MS;
        truth.false_start_t = fsT;
      }
      if (behavior === 'tilt') {
        truth.onset_t = startT + TILT_START_MS;
        truth.end_t = startT + TILT_START_MS + 2 * TILT_MOVE_MS + TILT_HOLD_MS;
      }
      try {
        if (mode === 'offline') runOffline(listener);
        else runRealtime(listener);
      } catch (e) {
        source.stop();
        return Promise.reject(e);
      }
      return Promise.resolve();
    },
    stop(): void {
      stopped = true;
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    },
    notifyCue(cue: CueKind, t: number): void {
      if (cue === 'go') {
        if (truth.go_t !== null) return; // only the first GO counts
        truth.go_t = t;
        if (fsT !== null && fsT > t) {
          // GO came before the participant got impatient: no false start
          fsT = null;
          truth.false_start_t = null;
        }
        if (behavior === 'normal' || behavior === 'false_start' || behavior === 'too_slow') {
          onsetT = t + reactionMs;
          arriveT = onsetT + moveMs;
          truth.onset_t = onsetT;
          truth.end_t = arriveT;
        }
      } else if (cue === 'end') {
        if (returnT !== null || arriveT === null) return;
        returnT = Math.max(t + returnDelayMs, arriveT + 150);
        truth.return_t = returnT;
      }
    },
  };
  return source;
}
