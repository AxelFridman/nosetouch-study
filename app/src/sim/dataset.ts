/**
 * Synthetic dataset generation through the REAL recording pipeline. DOM-free: runs in Node
 * (scripts/synth.ts, scripts/evalSegmentation.ts, tests) as well as in a browser.
 *
 * Per simulated participant i (one session each):
 *  - design (planDatasetDesign): platform ios for even i, android for odd i; round(N·leftFraction)
 *    left-handers, picked alternately from the two platform groups; sexes balanced female/male;
 *    the protocol seed's parity (→ protocol.resolveFirstHand) flips every two participants, so the
 *    first hand is crossed with the platform.
 *  - SimParticipant from model.randomSimParticipant (age, height → arm length, movement style) and
 *    SimPlatform from defaultSimPlatform: ios → gravitySign opts.iosGravitySign (default
 *    IOS_GRAVITY_SIGN = −1), engine 'webkit', full-precision values; android → +1, 'blink',
 *    Chromium quantisation (platform.quantize).
 *  - ParticipantProfile with EHI-SF answers drawn consistent with the handedness and scored with
 *    util.scoreEhi; the session holds config.redactProfile's snapshot.
 *  - onboarding as the app runs it, on a virtual clock: sensor check = sensors.measureRates on a
 *    simulated source (2.5 s) → capabilities.measured; calibration = calibration.poseStats of a
 *    'static' flat_up window (lead 3 s, 2 s) and a 'no_move' upright_toward window (lead 2 s, 2 s)
 *    → buildCalibration with axis null (the simulator performs no tilt step; same note as the
 *    calibration screen in simulation mode).
 *  - plan = protocol.buildPlan (a custom spec goes through normalizeCustomSpec) → flattenPlan; every
 *    attempt runs createSimulatedSource (offline) → createTrialController with the options
 *    sessionFlow.buildTrialOptions builds (resolveParams of the session's base params, the session
 *    calibration, foreperiod rng makeRng(trialSeed(session seed, trial_index, attempt))) and a no-op
 *    CueSink.
 *  - attempts as SessionScreen + sessionFlow.saveAttempt handle them: status 'complete' → accepted
 *    (the simulated participant always taps Next) and the session moves on; any other status → saved
 *    as a failed attempt (accepted false) and the same planned trial runs again with attempt + 1.
 *    An attempt is performed with behaviour 'no_move' (→ no_movement) with probability
 *    failureRate/2, 'false_start' (anticipatory twitch before GO → false_starts ≥ 1, normally still
 *    complete) with failureRate/2, else 'normal'. After maxAttemptsPerTrial failed attempts of one
 *    planned trial the session ends there as 'abandoned' (does not happen with the defaults).
 *  - progress / status / timestamps follow the sessionFlow rules; simulated: true everywhere.
 * Deterministic for given options: every random choice comes from util.makeRng(mixSeed(seed, …))
 * streams (one per participant and purpose, one per attempt), ids are seeded UUIDs and every time
 * comes from a virtual clock starting at opts.startEpochMs (never Date.now()).
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { buildCalibration, poseStats } from '../core/calibration';
import { redactProfile } from '../core/config';
import {
  APP_VERSION,
  DEFAULT_SEGMENTATION_PARAMS,
  DEFAULT_STUDY_CONFIG,
  PROTOCOL_VERSION,
  SCHEMA_VERSION,
} from '../core/constants';
import {
  buildPlan,
  DEFAULT_CUSTOM_SPEC,
  estimateMinutes,
  flattenPlan,
  normalizeCustomSpec,
  resolveFirstHand,
  resolveParams,
} from '../core/protocol';
import { measureRates, parseUserAgent } from '../core/sensors';
import { createTrialController } from '../core/trialController';
import type {
  CalibrationResult,
  CueSink,
  CustomProtocolSpec,
  DeviceInfo,
  EhiAnswer,
  EhiShortForm,
  Hand,
  Language,
  MotionSample,
  ParticipantProfile,
  PhoneHand,
  PlannedTrial,
  Posture,
  ProtocolOptions,
  ProtocolPreset,
  SegmentationParams,
  SensorCapabilities,
  SensorSource,
  SessionRecord,
  Sex,
  TrialCondition,
  TrialRecord,
  TrialStatus,
} from '../core/types';
import { isoTime, makeRng, mixSeed, scoreEhi, shuffle } from '../core/util';
import { IOS_GRAVITY_SIGN, defaultSimPlatform, randomSimParticipant, type SimParticipant, type SimPlatform } from './model';
import { createSimulatedSource, type SimBehavior, type SimTruth } from './simulatedSource';

// ───────────────────────────── options ─────────────────────────────

export const SYNTH_STUDY_CODE = 'synthetic-v1';
/** 2026-10-05T07:00:00Z: the first participant's page load (later participants: one day apart). */
export const SYNTH_START_EPOCH_MS = Date.UTC(2026, 9, 5, 7, 0, 0);

export interface DatasetOptions {
  /** simulated participants, one session each (default 12) */
  participants?: number;
  preset?: ProtocolPreset; // default 'standard'
  /** spec for preset 'custom' (normalizeCustomSpec; default DEFAULT_CUSTOM_SPEC); ignored otherwise */
  custom?: CustomProtocolSpec | null;
  seed?: number; // uint32, default 42
  /** share of left-handed participants: exactly round(participants · leftFraction) (default 0.3) */
  leftFraction?: number;
  /** probability that an attempt is a 'no_move' or a 'false_start' (half each); default 0.03 */
  failureRate?: number;
  studyCode?: string; // default SYNTH_STUDY_CODE
  /** gravitySign of the simulated iOS devices (default IOS_GRAVITY_SIGN) */
  iosGravitySign?: 1 | -1;
  posture?: Posture; // default 'standing'
  /** base params stored in session.settings; trials get resolveParams(base, condition) */
  segmentationParams?: SegmentationParams;
  /** epoch ms of participant 0's page load (default SYNTH_START_EPOCH_MS) */
  startEpochMs?: number;
  /** session.timezone_offset_min (minutes east of UTC, default 0) */
  timezoneOffsetMin?: number;
  language?: Language; // session.settings.language, default 'en'
  /** failed attempts of one planned trial after which the session is abandoned (default 8) */
  maxAttemptsPerTrial?: number;
}

export interface ResolvedDatasetOptions {
  participants: number;
  preset: ProtocolPreset;
  custom: CustomProtocolSpec | null;
  seed: number;
  leftFraction: number;
  failureRate: number;
  studyCode: string;
  iosGravitySign: 1 | -1;
  posture: Posture;
  segmentationParams: SegmentationParams;
  startEpochMs: number;
  timezoneOffsetMin: number;
  language: Language;
  maxAttemptsPerTrial: number;
}

const STUDY_CODE_RE = /^[A-Za-z0-9_-]{1,64}$/; // schema $defs.Id
const PRESETS: readonly ProtocolPreset[] = ['quick', 'standard', 'full', 'custom'];

function fraction(name: string, v: number, maxExclusive = false): number {
  if (!Number.isFinite(v) || v < 0 || v > 1 || (maxExclusive && v === 1)) {
    throw new Error(`generateDataset: ${name} must be in [0, 1${maxExclusive ? ')' : ']'}, got ${v}`);
  }
  return v;
}

/** Fills the defaults and rejects values that would produce invalid records. */
export function resolveDatasetOptions(opts: DatasetOptions = {}): ResolvedDatasetOptions {
  const participants = opts.participants ?? 12;
  if (!Number.isInteger(participants) || participants < 1) {
    throw new Error(`generateDataset: participants must be a positive integer, got ${participants}`);
  }
  const preset = opts.preset ?? 'standard';
  if (!PRESETS.includes(preset)) throw new Error(`generateDataset: unknown preset "${String(preset)}"`);
  const studyCode = opts.studyCode ?? SYNTH_STUDY_CODE;
  if (!STUDY_CODE_RE.test(studyCode)) throw new Error(`generateDataset: study code must match ${STUDY_CODE_RE}`);
  const iosGravitySign = opts.iosGravitySign ?? IOS_GRAVITY_SIGN;
  if (iosGravitySign !== 1 && iosGravitySign !== -1) throw new Error('generateDataset: iosGravitySign must be 1 or -1');
  const maxAttemptsPerTrial = opts.maxAttemptsPerTrial ?? 8;
  if (!Number.isInteger(maxAttemptsPerTrial) || maxAttemptsPerTrial < 1) {
    throw new Error('generateDataset: maxAttemptsPerTrial must be a positive integer');
  }
  const startEpochMs = opts.startEpochMs ?? SYNTH_START_EPOCH_MS;
  if (!Number.isFinite(startEpochMs)) throw new Error('generateDataset: startEpochMs must be finite');
  const timezoneOffsetMin = opts.timezoneOffsetMin ?? 0;
  if (!Number.isInteger(timezoneOffsetMin)) throw new Error('generateDataset: timezoneOffsetMin must be an integer');
  return {
    participants,
    preset,
    // normalised here, so the stored spec describes the plan exactly (protocol.normalizeCustomSpec)
    custom: preset === 'custom' ? normalizeCustomSpec(opts.custom ?? DEFAULT_CUSTOM_SPEC) : null,
    seed: (opts.seed ?? 42) >>> 0,
    leftFraction: fraction('leftFraction', opts.leftFraction ?? 0.3),
    // 1 would make every attempt fail and every session abandoned
    failureRate: fraction('failureRate', opts.failureRate ?? 0.03, true),
    studyCode,
    iosGravitySign,
    posture: opts.posture ?? 'standing',
    segmentationParams: { ...(opts.segmentationParams ?? DEFAULT_SEGMENTATION_PARAMS) },
    startEpochMs,
    timezoneOffsetMin,
    language: opts.language ?? 'en',
    maxAttemptsPerTrial,
  };
}

// ───────────────────────────── seeds ─────────────────────────────

/** Salts of the independent random streams (any distinct constants). */
const SALT = {
  design: 0x44534731,
  sim: 0x53494d31,
  platform: 0x504c4131,
  profile: 0x50524f31,
  session: 0x53455331,
  clock: 0x434c4b31,
  calibration: 0x43414c31,
  attempt: 0x41545431,
} as const;

function seedOf(seed: number, ...parts: number[]): number {
  return parts.reduce((h, p) => mixSeed(h, p), seed >>> 0);
}

/**
 * Copy of app/sessionFlow.trialSeed (src/app is not DOM-free, so it cannot be imported here); the
 * foreperiod rng of an attempt is makeRng(appTrialSeed(session seed, trial_index, attempt)) in the
 * app, so a recorded attempt can be re-simulated exactly. tests/synth.test.ts checks they agree.
 */
export function appTrialSeed(sessionSeed: number, trialIndex: number, attempt: number): number {
  let h = (sessionSeed ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (trialIndex + 1), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h ^ (attempt * 0x27d4eb2f), 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** RFC 4122 v4 layout from a seeded rng (same shape as util.newId, but reproducible). */
export function seededUuid(rng: () => number): string {
  const b: number[] = [];
  for (let i = 0; i < 16; i++) b.push(Math.floor(rng() * 256) & 0xff);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const hex = b.map((x) => x.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function uint32(rng: () => number): number {
  return Math.floor(rng() * 4294967296) >>> 0;
}

function pickWeighted<T>(rng: () => number, table: readonly (readonly [T, number])[]): T {
  const total = table.reduce((s, [, w]) => s + w, 0);
  let u = rng() * total;
  for (const [v, w] of table) {
    if (u < w) return v;
    u -= w;
  }
  return table[table.length - 1][0];
}

function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.min(items.length - 1, Math.floor(rng() * items.length))];
}

// ───────────────────────────── design ─────────────────────────────

export interface ParticipantDesign {
  index: number;
  platform: 'ios' | 'android';
  dominant: Hand;
  sex: Sex;
  /** parity of the protocol seed: 0 → right hand first, 1 → left first (protocol.resolveFirstHand) */
  seedParity: 0 | 1;
}

/** The between-participant design (cheap: no simulation). */
export function planDatasetDesign(opts: DatasetOptions = {}): ParticipantDesign[] {
  const o = resolveDatasetOptions(opts);
  const n = o.participants;
  const rng = makeRng(seedOf(o.seed, SALT.design));
  const platformOf = (i: number): 'ios' | 'android' => (i % 2 === 0 ? 'ios' : 'android');
  const all = Array.from({ length: n }, (_, i) => i);
  const groups = [shuffle(all.filter((i) => platformOf(i) === 'ios'), rng), shuffle(all.filter((i) => platformOf(i) === 'android'), rng)];
  // left-handers alternate between the platform groups so handedness is not confounded with platform
  const nLeft = Math.round(n * o.leftFraction);
  const left = new Set<number>();
  for (let k = rng() < 0.5 ? 0 : 1; left.size < nLeft; k++) {
    const next = groups[k % 2].shift();
    if (next !== undefined) left.add(next);
  }
  const sexes = shuffle(
    all.map((i): Sex => (i % 2 === 0 ? 'female' : 'male')),
    rng,
  );
  return all.map((i) => ({
    index: i,
    platform: platformOf(i),
    dominant: left.has(i) ? 'left' : 'right',
    sex: sexes[i],
    seedParity: (Math.floor(i / 2) % 2) as 0 | 1,
  }));
}

// ───────────────────────────── participant profile ─────────────────────────────

type EhiTable = Record<keyof EhiShortForm, readonly (readonly [EhiAnswer, number])[]>;

/** Answer distributions per item: writing is the most lateralised item, throwing the least. */
const EHI_RIGHT_HANDED: EhiTable = {
  writing: [['always_right', 0.93], ['usually_right', 0.07]],
  throwing: [['always_right', 0.78], ['usually_right', 0.14], ['both', 0.05], ['usually_left', 0.03]],
  toothbrush: [['always_right', 0.66], ['usually_right', 0.25], ['both', 0.09]],
  spoon: [['always_right', 0.72], ['usually_right', 0.21], ['both', 0.07]],
};

/** Left-handers are less consistent (many throw or eat with the right hand). */
const EHI_LEFT_HANDED: EhiTable = {
  writing: [['always_left', 0.9], ['usually_left', 0.1]],
  throwing: [['always_left', 0.5], ['usually_left', 0.2], ['both', 0.1], ['usually_right', 0.12], ['always_right', 0.08]],
  toothbrush: [['always_left', 0.55], ['usually_left', 0.27], ['both', 0.11], ['usually_right', 0.07]],
  spoon: [['always_left', 0.5], ['usually_left', 0.29], ['both', 0.12], ['usually_right', 0.09]],
};

const PHONE_HAND: Record<Hand, readonly (readonly [PhoneHand, number])[]> = {
  right: [['right', 0.75], ['both', 0.17], ['left', 0.08]],
  left: [['left', 0.5], ['both', 0.25], ['right', 0.25]],
};

interface PhoneSpec {
  model: string; // what a participant would type
  hint: string | null; // UA-CH model (Android only)
  w: number;
  h: number;
  dpr: number;
  memory: number | null; // navigator.deviceMemory (Chromium only, capped at 8)
  cores: number;
}

const IOS_PHONES: readonly PhoneSpec[] = [
  { model: 'iPhone 13', hint: null, w: 390, h: 844, dpr: 3, memory: null, cores: 6 },
  { model: 'iPhone 14', hint: null, w: 390, h: 844, dpr: 3, memory: null, cores: 6 },
  { model: 'iPhone 15', hint: null, w: 393, h: 852, dpr: 3, memory: null, cores: 6 },
  { model: 'iPhone 15 Pro Max', hint: null, w: 430, h: 932, dpr: 3, memory: null, cores: 6 },
  { model: 'iPhone 16', hint: null, w: 393, h: 852, dpr: 3, memory: null, cores: 6 },
  { model: 'iPhone SE', hint: null, w: 375, h: 667, dpr: 2, memory: null, cores: 6 },
];

const ANDROID_PHONES: readonly PhoneSpec[] = [
  { model: 'Pixel 8', hint: 'Pixel 8', w: 412, h: 915, dpr: 2.625, memory: 8, cores: 9 },
  { model: 'Pixel 7a', hint: 'Pixel 7a', w: 412, h: 915, dpr: 2.625, memory: 8, cores: 8 },
  { model: 'Galaxy S24', hint: 'SM-S921B', w: 360, h: 780, dpr: 3, memory: 8, cores: 10 },
  { model: 'Galaxy A54', hint: 'SM-A546B', w: 412, h: 915, dpr: 2.625, memory: 8, cores: 8 },
  { model: 'Galaxy A15', hint: 'SM-A155F', w: 384, h: 854, dpr: 2.8125, memory: 4, cores: 8 },
  { model: 'Redmi Note 13', hint: '23129RAA4G', w: 393, h: 873, dpr: 2.75, memory: 8, cores: 8 },
];

/** Safari ≥ 26 freezes "iPhone OS 18_6" in its UA; its own version is the iOS version. */
const IOS_SAFARI_VERSIONS = ['26.6', '27.0'] as const;
/** Reduced Chrome UA (real version/model come from UA Client Hints, as getDeviceInfo reads them). */
const CHROME_VERSIONS = ['151', '152', '153'] as const;
const ANDROID_VERSIONS = ['14.0.0', '15.0.0', '16.0.0'] as const;

function buildDevice(platform: 'ios' | 'android', rng: () => number, language: Language, timeOrigin: number): {
  device: DeviceInfo;
  spec: PhoneSpec;
} {
  const ios = platform === 'ios';
  const spec = pick(rng, ios ? IOS_PHONES : ANDROID_PHONES);
  const ua = ios
    ? `Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/${pick(rng, IOS_SAFARI_VERSIONS)} Mobile/15E148 Safari/604.1`
    : `Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${pick(rng, CHROME_VERSIONS)}.0.0.0 Mobile Safari/537.36`;
  const parsed = parseUserAgent(ua, { maxTouchPoints: 5 });
  const uaChVersion = ios ? null : pick(rng, ANDROID_VERSIONS);
  return {
    spec,
    device: {
      user_agent: ua,
      platform: parsed.platform,
      browser: parsed.browser,
      os_version: uaChVersion ?? parsed.os_version,
      model_hint: spec.hint ?? parsed.model_hint,
      standalone: false,
      screen_w: spec.w,
      screen_h: spec.h,
      dpr: spec.dpr,
      language: language === 'he' ? 'he-IL' : 'en-US',
      hardware_concurrency: spec.cores,
      device_memory_gb: spec.memory,
      time_origin_ms: timeOrigin,
      engine: ios ? 'webkit' : 'blink',
      in_app_browser: parsed.in_app_browser,
    },
  };
}

function buildProfile(sim: SimParticipant, rng: () => number, phoneModel: string, createdAt: string): ParticipantProfile {
  const table = sim.dominant === 'left' ? EHI_LEFT_HANDED : EHI_RIGHT_HANDED;
  const ehi: EhiShortForm = {
    writing: pickWeighted(rng, table.writing),
    throwing: pickWeighted(rng, table.throwing),
    toothbrush: pickWeighted(rng, table.toothbrush),
    spoon: pickWeighted(rng, table.spoon),
  };
  const { lq, cls } = scoreEhi(ehi);
  const phoneHand = pickWeighted(rng, PHONE_HAND[sim.dominant]);
  const modelKnown = rng() < 0.85; // the field is optional in the profile form
  const phoneCase = rng() < 0.7;
  return {
    participant_id: sim.participant_id,
    nickname: null,
    age_years: sim.age_years,
    sex: sim.sex,
    handedness_self: sim.handedness,
    ehi_sf: ehi,
    ehi_lq: lq,
    ehi_class: cls,
    phone_hand: phoneHand,
    height_cm: sim.height_cm,
    arm_issue: false,
    phone_model: modelKnown ? phoneModel : null,
    phone_case: phoneCase,
    created_at: createdAt,
    updated_at: createdAt,
  };
}

// ───────────────────────────── recording helpers ─────────────────────────────

/** Calibration screen windows (CalibrationScreen.STEP_TIMING) and the sensor check (MEASURE_MS). */
const CAL_FLAT = { leadMs: 3000, durationMs: 2000 };
const CAL_UPRIGHT = { leadMs: 2000, durationMs: 2000 };
const SENSOR_CHECK_MS = 2500;
/** CalibrationScreen adds this note in simulation mode (no tilt step → axis_check null). */
const SIM_CALIBRATION_NOTE = 'simulated: tilt step not performed by the simulator';

/** Offline counterpart of CalibrationScreen.recordWindow: motion samples of [lead, lead + duration] (source clock). */
async function recordWindow(source: SensorSource, leadMs: number, durationMs: number): Promise<MotionSample[]> {
  const samples: MotionSample[] = [];
  const from = source.clock.now() + leadMs;
  const to = from + durationMs;
  await source.start({
    onMotion(s) {
      if (s.t >= from && s.t <= to) samples.push(s);
      if (s.t >= to) source.stop();
    },
    onOrientation() {},
  });
  return samples;
}

const SILENT_CUES: CueSink = { play() {} };
const PORTRAIT = (): { type: string | null; angle: number | null } => ({ type: 'portrait-primary', angle: 0 });
const NOT_HIDDEN = (): boolean => false;

function holdCondition(hand: Hand, orientation: TrialCondition['orientation'], posture: Posture): TrialCondition {
  return { hand, orientation, speed: 'natural', start_position: 'natural', variant: 'standard', posture, rep: 0 };
}

// ───────────────────────────── generation ─────────────────────────────

/** One attempt as the simulator performed it (truth in the SOURCE clock; subtract t0 for record time). */
export interface AttemptInfo {
  trial_id: string;
  trial_index: number;
  attempt: number;
  behavior: SimBehavior;
  /** = timing.t0_perf_ms before rounding: source-clock time of the record's t = 0 */
  t0: number;
  truth: SimTruth;
}

export interface SyntheticSession {
  design: ParticipantDesign;
  sim: { participant: SimParticipant; platform: SimPlatform };
  /** the participant's own profile (the session holds the redacted snapshot) */
  profile: ParticipantProfile;
  session: SessionRecord;
  /** every attempt in recording order (what the backend stores as trial_*.json) */
  trials: TrialRecord[];
  /** parallel to trials */
  attempts: AttemptInfo[];
}

export interface SyntheticDataset {
  options: ResolvedDatasetOptions;
  sessions: SyntheticSession[];
}

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

async function generateSession(o: ResolvedDatasetOptions, d: ParticipantDesign, usedIds: Set<string>): Promise<SyntheticSession> {
  const i = d.index;
  const stream = (salt: number) => makeRng(seedOf(o.seed, salt, i));

  // ── simulated person and phone ──
  const simRng = stream(SALT.sim);
  let participant = randomSimParticipant(simRng, { handedness: d.dominant, dominant: d.dominant, sex: d.sex });
  while (usedIds.has(participant.participant_id)) {
    // 32^6 ids: a collision is rare, the redraw keeps determinism
    participant = { ...participant, participant_id: randomSimParticipant(simRng).participant_id };
  }
  usedIds.add(participant.participant_id);
  const base = defaultSimPlatform(d.platform, stream(SALT.platform));
  const platform: SimPlatform = d.platform === 'ios' ? { ...base, gravitySign: o.iosGravitySign } : base;

  // ── virtual clock: performance.now()-like `perf` since page load, epoch = timeOrigin + perf ──
  const clk = stream(SALT.clock);
  const between = (lo: number, hi: number) => lo + (hi - lo) * clk();
  const timeOrigin = Math.round((o.startEpochMs + i * DAY_MS + between(0, 8 * HOUR_MS)) * 1000) / 1000;
  let perf = between(400, 1500); // app booted
  const at = (p: number) => isoTime(timeOrigin + p);

  perf += between(15_000, 60_000);
  const consentAt = at(perf);
  perf += between(45_000, 150_000);
  const profRng = stream(SALT.profile);
  const { device, spec } = buildDevice(d.platform, profRng, o.language, timeOrigin);
  const profile = buildProfile(participant, profRng, spec.model, at(perf));

  // ── sensor check and calibration (simulated sources on the same clock) ──
  const calRng = stream(SALT.calibration);
  const source = (condition: TrialCondition, behavior: SimBehavior, startTime: number): SensorSource =>
    createSimulatedSource({
      mode: 'offline',
      participant,
      platform,
      condition,
      seed: uint32(calRng),
      behavior,
      startTime,
      baseEpochMs: timeOrigin,
    });
  perf += between(10_000, 30_000);
  const measured = await measureRates(source(holdCondition(participant.dominant, 'upright_toward', o.posture), 'no_move', perf), SENSOR_CHECK_MS);
  perf += SENSOR_CHECK_MS;
  const capabilities: SensorCapabilities = {
    secure_context: true,
    device_motion: true,
    device_orientation: true,
    permission_api: true, // iOS 13+, and Chrome 151+ exposes requestPermission too (sensors.ts)
    permission_state: 'granted',
    generic_relative_orientation: d.platform === 'android',
    wake_lock: true,
    vibrate: d.platform === 'android',
    share_files: true,
    measured,
  };
  perf += between(15_000, 40_000);
  const flatSamples = await recordWindow(source(holdCondition(participant.dominant, 'flat_up', o.posture), 'static', perf), CAL_FLAT.leadMs, CAL_FLAT.durationMs);
  perf += CAL_FLAT.leadMs + CAL_FLAT.durationMs + between(5_000, 15_000);
  const uprightSamples = await recordWindow(
    source(holdCondition(participant.dominant, 'upright_toward', o.posture), 'no_move', perf),
    CAL_UPRIGHT.leadMs,
    CAL_UPRIGHT.durationMs,
  );
  perf += CAL_UPRIGHT.leadMs + CAL_UPRIGHT.durationMs;
  const calibration: CalibrationResult = buildCalibration({
    performedAt: at(perf),
    flat: poseStats(flatSamples),
    upright: poseStats(uprightSamples),
    axis: null,
    notes: [SIM_CALIBRATION_NOTE],
  });

  // ── session ──
  const sesRng = stream(SALT.session);
  const sessionId = seededUuid(sesRng);
  const sessionSeed = ((uint32(sesRng) & 0xfffffffe) | d.seedParity) >>> 0;
  const protocolOpts: ProtocolOptions = { preset: o.preset, seed: sessionSeed, posture: o.posture, custom: o.custom };
  const plan = buildPlan(protocolOpts);
  const flat = flattenPlan(plan);
  perf += between(10_000, 40_000); // setup screen
  const startedAt = at(perf);
  const config = { ...DEFAULT_STUDY_CONFIG, study_code: o.studyCode };

  const trials: TrialRecord[] = [];
  const attempts: AttemptInfo[] = [];
  const progress = { next_trial_index: 0, n_attempts: 0, n_accepted: 0 };
  let status: SessionRecord['status'] = 'completed';
  let lastBlock: number | null = null;
  perf += between(5_000, 15_000); // first block intro

  const runAttempt = async (planned: PlannedTrial, attempt: number): Promise<TrialRecord> => {
    const aRng = makeRng(seedOf(o.seed, SALT.attempt, i, planned.trial_index, attempt));
    const trialId = seededUuid(aRng);
    const u = aRng();
    const behavior: SimBehavior = u < o.failureRate / 2 ? 'no_move' : u < o.failureRate ? 'false_start' : 'normal';
    perf += 1000 + 2000 * aRng(); // reads the instruction, taps Ready
    const src = createSimulatedSource({
      mode: 'offline',
      participant,
      platform,
      condition: planned.condition,
      seed: uint32(aRng),
      behavior,
      startTime: perf,
      baseEpochMs: timeOrigin,
    });
    // = sessionFlow.buildTrialOptions
    const ctrl = createTrialController({
      source: src,
      cues: SILENT_CUES,
      ids: {
        trial_id: trialId,
        session_id: sessionId,
        participant_id: participant.participant_id,
        study_code: o.studyCode,
        block_index: planned.block_index,
        trial_index: planned.trial_index,
        trial_in_block: planned.trial_in_block,
        attempt,
      },
      condition: planned.condition,
      params: resolveParams(o.segmentationParams, planned.condition),
      calibration,
      rng: makeRng(appTrialSeed(sessionSeed, planned.trial_index, attempt)),
      appVersion: APP_VERSION,
      simulated: src.kind === 'simulated',
      getScreenOrientation: PORTRAIT,
      isHidden: NOT_HIDDEN,
    });
    const t0 = src.clock.now();
    await ctrl.start();
    const raw = await ctrl.result;
    // sessionFlow.applyDecision(accept) for a complete trial, asFailedAttempt(record, null) otherwise
    const accepted = raw.outcome.status === 'complete';
    const record: TrialRecord = { ...raw, outcome: { ...raw.outcome, accepted, participant_flag: null, note: null } };
    attempts.push({ trial_id: trialId, trial_index: planned.trial_index, attempt, behavior, t0, truth: { ...src.truth } });
    // review (complete) or the failure notice, then the decision tap
    perf = t0 + raw.timing.stop_ms + (accepted ? 2000 + 4000 * aRng() : 2500 + 2500 * aRng());
    return record;
  };

  outer: for (const planned of flat) {
    if (lastBlock !== null && planned.block_index !== lastBlock) perf += between(15_000, 45_000); // break + intro
    lastBlock = planned.block_index;
    for (let attempt = 1; ; attempt++) {
      const record = await runAttempt(planned, attempt);
      trials.push(record);
      // = sessionFlow.saveAttempt for a new attempt
      progress.n_attempts += 1;
      if (record.outcome.accepted) {
        progress.n_accepted += 1;
        progress.next_trial_index = Math.min(flat.length, Math.max(progress.next_trial_index, planned.trial_index + 1));
        break;
      }
      if (attempt >= o.maxAttemptsPerTrial) {
        status = 'abandoned';
        break outer;
      }
    }
  }
  const endedAt = at(perf);

  const session: SessionRecord = {
    record_type: 'session',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    session_id: sessionId,
    participant_id: participant.participant_id,
    study_code: o.studyCode,
    started_at: startedAt,
    updated_at: endedAt,
    ended_at: endedAt,
    status,
    participant: redactProfile(profile, config),
    device,
    capabilities,
    calibration,
    protocol: {
      preset: o.preset,
      protocol_version: PROTOCOL_VERSION,
      seed: sessionSeed,
      posture: o.posture,
      first_hand: resolveFirstHand(protocolOpts),
      n_planned_trials: flat.length,
      estimated_minutes: estimateMinutes(plan),
      custom: o.custom ? { ...o.custom } : null,
      plan,
    },
    settings: {
      language: o.language,
      audio: true,
      speech: true,
      segmentation_params: { ...o.segmentationParams },
    },
    progress,
    consent: { version: config.consent_version, accepted_at: consentAt },
    timezone_offset_min: o.timezoneOffsetMin,
    simulated: true,
    notes: `synthetic (sim/dataset.ts): seed ${o.seed}, participant ${i + 1} of ${o.participants}, ${d.platform}-like, dominant hand ${d.dominant}`,
  };
  return { design: d, sim: { participant, platform }, profile, session, trials, attempts };
}

/** Yields one participant's session at a time (keeps memory flat for large datasets). */
export async function* iterateDataset(opts: DatasetOptions = {}): AsyncGenerator<SyntheticSession> {
  const o = resolveDatasetOptions(opts);
  const design = planDatasetDesign(o);
  const usedIds = new Set<string>();
  for (const d of design) yield await generateSession(o, d, usedIds);
}

/** The whole dataset in memory (tests, small runs); scripts/synth.ts streams with iterateDataset. */
export async function generateDataset(opts: DatasetOptions = {}): Promise<SyntheticDataset> {
  const sessions: SyntheticSession[] = [];
  for await (const s of iterateDataset(opts)) sessions.push(s);
  return { options: resolveDatasetOptions(opts), sessions };
}

// ───────────────────────────── summaries ─────────────────────────────

export interface SessionSummary {
  participant_id: string;
  session_id: string;
  platform: 'ios' | 'android';
  dominant: Hand;
  sex: Sex;
  age_years: number | null;
  height_cm: number | null;
  ehi_class: ParticipantProfile['ehi_class'];
  first_hand: Hand;
  status: SessionRecord['status'];
  n_planned: number;
  n_accepted: number;
  n_attempts: number;
  statuses: Record<TrialStatus, number>;
  false_starts: number;
  /** attempts the simulator performed with each behaviour */
  behaviors: Partial<Record<SimBehavior, number>>;
  gravity_sign: 1 | -1 | null;
}

export function summarizeSession(s: SyntheticSession): SessionSummary {
  const statuses: Record<TrialStatus, number> = { complete: 0, no_movement: 0, too_long: 0, aborted: 0, interrupted: 0 };
  let falseStarts = 0;
  for (const t of s.trials) {
    statuses[t.outcome.status] += 1;
    falseStarts += t.timing.false_starts;
  }
  const behaviors: Partial<Record<SimBehavior, number>> = {};
  for (const a of s.attempts) behaviors[a.behavior] = (behaviors[a.behavior] ?? 0) + 1;
  return {
    participant_id: s.session.participant_id,
    session_id: s.session.session_id,
    platform: s.design.platform,
    dominant: s.design.dominant,
    sex: s.profile.sex,
    age_years: s.profile.age_years,
    height_cm: s.profile.height_cm,
    ehi_class: s.profile.ehi_class,
    first_hand: s.session.protocol.first_hand,
    status: s.session.status,
    n_planned: s.session.protocol.n_planned_trials,
    n_accepted: s.session.progress.n_accepted,
    n_attempts: s.session.progress.n_attempts,
    statuses,
    false_starts: falseStarts,
    behaviors,
    gravity_sign: s.session.calibration?.gravity_sign ?? null,
  };
}
