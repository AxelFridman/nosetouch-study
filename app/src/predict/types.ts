/**
 * Shared contracts of predict mode (Stage 3): the production page at ?predict=1.
 *
 * Spec: docs/stage3/APP_SPEC.md. Python ground truth (port it, do not redesign it):
 *   analysis/stage2/standardize.py      stream_from_record, canonical, time_normalised
 *   analysis/stage2/lanes/m4_detect.py  activity, find_segments, segment_table, rest_up_z
 *   analysis/stage2/lanes/m4_e2e.py     FittedPipeline.attempts (the attempt rule)
 *   analysis/stage2/lanes/m4_online.py  aggregate, stopping_rule
 *   analysis/stage2/lanes/m4_production.py  predict_stream (the cue-free production algorithm)
 *   analysis/stage2/lanes/final_infer.py    the final CNN (numpy reference of final_model.json)
 *
 * TYPE-ONLY MODULE: no runtime code, so importing it never adds bytes to any chunk. Every owner imports
 * from here; a change to this file is a contract change and goes through the orchestrator.
 *
 * Owners (docs/stage3/APP_SPEC.md section 6):
 *   pipeline/  LiveBuffer, RawMotion, StreamNormalization, Stream, Canonical, ApproachWindow
 *   detect/    Bout, SegmentRow, grip check, stopping rule, StreamPrediction, SessionEngine
 *   models/    ModelSpec, HandModel, ModelAttemptOutput, AggregationSpec, CalibratorJson, the JSON bundles
 *   UI         PredictParams, LiveSource, the screens (mount.tsx, PredictApp.tsx, components)
 *
 * Conventions (same as the collector, core/types.ts): time ms, acceleration m/s^2, angular velocity deg/s,
 * angles deg. Device frame (W3C): x right edge, y top edge, z out of the screen. P = P(right): y = 1 is the
 * RIGHT hand everywhere (stage2/config.py POSITIVE_HAND).
 */
import type { Hand, MotionSample, PermissionState, Vec3 } from '../core/types';

export type { Hand };

// ───────────────────────────── 1. URL parameters ─────────────────────────────

/** Short ids used in the URL (&model=<id>) and in every per-model record. */
export type ModelId = 'cnn' | 'm4' | 'phys6' | 'yaw';

export type PredictLang = 'en' | 'he';

/**
 * Parsed URL of predict mode. main.tsx decides predict mode itself (?predict=1 or ?mode=predict) BEFORE
 * the predict chunk loads; mount() parses the rest. Unknown or malformed values fall back to the default
 * and are listed in `ignored` (the page shows nothing about them; the debug panel does).
 */
export interface PredictParams {
  /** &model=<ModelId>: the model whose stopping rule ends the session (default 'cnn') */
  model: ModelId;
  /** &view=compare: open with the compare-all table expanded */
  compare: boolean;
  /** &sim=1: desktop demo, a synthetic stream replayed through the live path (never real sensors) */
  sim: boolean;
  /** &hand=left|right (sim only): the simulated hand; null = random per run */
  simHand: Hand | null;
  /** &seed=<uint32> (sim only): reproducible demo; null = random per run */
  simSeed: number | null;
  /** &lang=he: Hebrew, right to left (default en) */
  lang: PredictLang;
  /** &debug=1: attempt list, ignored movements, QC numbers, sample rate */
  debug: boolean;
  /** parameter names whose value was not understood */
  ignored: string[];
}

// ───────────────────────────── 2. live samples and their source ─────────────────────────────

/**
 * One devicemotion reading exactly as core/sensors.ts createBrowserSensorSource() delivers it:
 * t = performance.now() taken first thing in the handler (ms, source clock, NOT relative), ts =
 * event.timeStamp, acc = acceleration (gravity excluded, may be null), accg = accelerationIncludingGravity,
 * rr = rotationRate {alpha: about device x, beta: about y, gamma: about z} in deg/s, interval = raw
 * event.interval (Android ms, iOS SECONDS). Values are RAW: no sign normalisation (WebKit reports acc and
 * accg negated) and no rounding; LiveBuffer applies the recorder's rounding and the pipeline the sign.
 */
export type LiveSample = MotionSample;

/** core/sensors.ts getDeviceInfo().engine ('webkit' = every iOS browser, detected by webkitCompassHeading). */
export type BrowserEngine = 'webkit' | 'blink' | 'gecko' | 'unknown';

/** Why live sensing cannot run (the 'unsupported' screen shows one message per reason). */
export type SourceFailure =
  | 'insecure_context' // not https (core/sensors 'insecure_context')
  | 'not_supported' // no DeviceMotionEvent (desktop Safari/Firefox, old WebViews)
  | 'in_app_browser' // Instagram, Facebook, WhatsApp... (getDeviceInfo().in_app_browser)
  | 'permission_denied' // requestPermission() said 'denied' (iOS; Chrome 151+)
  | 'no_events' // no data-carrying devicemotion event within 1.5 s (desktop with no sensor)
  | 'no_gyroscope' // rotationRate null in every sample of the first second
  | 'no_accelerometer' // accelerationIncludingGravity null in every sample of the first second
  | 'engine_error'; // the engine threw while processing samples (never expected; shown instead of a frozen page)

/**
 * A stream of LiveSamples: the real sensors (live/browserSource.ts, wraps core/sensors.ts) or the desktop
 * demo (live/demoSource.ts, wraps sim/). Single-use like core SensorSource; stop() is idempotent.
 */
export interface LiveSource {
  readonly kind: 'browser' | 'demo';
  /** engine whose accelerometer sign the samples carry (the demo reports its simulated platform's) */
  readonly engine: BrowserEngine;
  /**
   * Starts delivery. The browser source's permission request happens BEFORE this, synchronously in the
   * Start tap (requestMotionPermission, core/sensors.ts); start() never prompts. Resolves once listening.
   */
  start(onSample: (s: LiveSample) => void, onFailure: (reason: SourceFailure) => void): Promise<void>;
  stop(): void;
}

/** What the UI learns before the first tap (no prompt): reasons that block the Start button. */
export interface Preflight {
  ok: boolean;
  failure: SourceFailure | null;
  engine: BrowserEngine;
  /** permission API present (iOS 13+, Chrome 151+): the Start tap must call requestMotionPermission() */
  needsPermission: boolean;
  permission: PermissionState;
  wakeLock: boolean;
  /** navigator.vibrate exists (Android): one 30 ms pulse per attempt; never sound */
  vibrate: boolean;
  /** device.platform === 'other' (a desktop): offer the demo link */
  desktop: boolean;
}

// ───────────────────────────── 3. pipeline: raw buffer → standardised stream → windows ─────────────────────────────

/** (n, 3) float64 rows, row-major (row i = data[3i], data[3i + 1], data[3i + 2]); NaN = missing. */
export interface Rows3 {
  readonly n: number;
  readonly data: Float64Array;
}

/**
 * The motion stream of one recording as stored in TrialRecord.streams.motion (motionlab.io.TrialStreams:
 * t, acc, accg, rr). A live buffer and a recorded trial MUST produce identical RawMotion for identical
 * samples: LiveBuffer rounds exactly like core/recorder.ts StreamRecorder.addMotionRel (t to 0.01 ms
 * relative to the first sample, acc/accg 1e-4, rr 1e-3, each missing component NaN on its own, as the
 * recorder stores it; core/sensors.ts already reports a vector with a missing component as null, so in
 * practice whole vectors are NaN).
 */
export interface RawMotion {
  /** handler time, ms since t0 (the first live sample, or the trial's t0 for a record) */
  readonly t: Float64Array;
  /** acceleration excluding gravity, raw device frame, RAW sign */
  readonly acc: Rows3;
  /** accelerationIncludingGravity, raw device frame, RAW sign */
  readonly accg: Rows3;
  /** rotationRate in schema order [alpha, beta, gamma], deg/s (before the axis mapping) */
  readonly rr: Rows3;
}

/** motionlab.conventions RR_MAPPINGS keys. Production assumes 'w3c' (every current browser). */
export type RateMappingVerdict = 'w3c' | 'w3c_inverted' | 'legacy' | 'legacy_inverted';

/**
 * The norm_v2 subset that stream_from_record reads (motionlab.conventions.Normalization):
 * acc and accg are multiplied by gravitySign, rr is mapped by (rrVerdict, rrUnits), and the repeated-sample
 * rule depends on the engine. Production has no calibration: gravitySign = the engine prior (webkit -1,
 * blink/gecko +1, unknown +1), W3C deg/s, no gravity magnitude, no baseline gravity, no iOS quantisation.
 * Fixtures from records carry the session's resolved Normalization instead (source 'record').
 */
export interface StreamNormalization {
  readonly gravitySign: 1 | -1;
  readonly rrVerdict: RateMappingVerdict;
  readonly rrUnits: 'deg_per_s' | 'rad_per_s';
  readonly engine: BrowserEngine | null;
  /** stage2 config QUANTIZE_IOS (false in every exported model) */
  readonly quantizeIos: boolean;
  readonly source: 'engine_prior' | 'record';
}

/**
 * standardize.Stream: the recording on the uniform 60 Hz grid (t_raw[0] + k * 1000/60, through t_raw[-1]),
 * sign-normalised, linearly resampled (gaps > 100 ms stay NaN) and zero-phase low-passed (Butterworth
 * order 4, 15 Hz, scipy sosfiltfilt with padlen min(15, run - 1), run by run over finite samples).
 */
export interface Stream {
  readonly t: Float64Array;
  /** specific force f = accg with the W3C sign (up at rest) */
  readonly f: Rows3;
  /** the browser's linear acceleration (QC only; NaN when missing) */
  readonly acc: Rows3;
  /** angular velocity about device x, y, z (deg/s) */
  readonly gyro: Rows3;
  /** 1 where f and gyro were measured (0 inside gaps > 100 ms) */
  readonly valid: Uint8Array;
  readonly fsHz: number;
  readonly engine: BrowserEngine | null;
  readonly repeatsDropped: number;
  /** raw sample times (ms), for gap QC */
  readonly tRaw: Float64Array;
  /** always null on the production path (m4_detect.as_stream drops them); set only for oracle parity tests */
  readonly gravityMagnitude: number | null;
  readonly baselineGravity: Vec3 | null;
}

/** standardize.CHANNELS, in this order (row index of every (17, T) array). */
export type CanonicalChannel =
  | 'acc_x'
  | 'acc_y'
  | 'acc_z'
  | 'gyro_x'
  | 'gyro_y'
  | 'gyro_z'
  | 'up_x'
  | 'up_y'
  | 'up_z'
  | 'gyro_mag'
  | 'acc_mag'
  | 'acc_vert'
  | 'acc_horiz'
  | 'yaw_rate'
  | 'gyro_horiz'
  | 'yaw_cum'
  | 'tilt_from_start';

/** standardize.Canonical: every channel on the (cropped) stream grid for one pair of bounds. */
export interface Canonical {
  readonly t: Float64Array;
  /** (17, n) float64, channel-major: X[c * n + i] */
  readonly X: Float64Array;
  readonly n: number;
  /** gyro-propagated gravity estimate (gravity_v1) */
  readonly g: Rows3;
  readonly valid: Uint8Array;
  readonly onsetMs: number;
  readonly endMs: number;
  readonly iOn: number;
  readonly iEnd: number;
  readonly closureDeg: number;
  readonly restPreN: number;
  readonly restPostN: number;
  readonly gravityMethod: string;
}

/**
 * standardize.time_normalised: the 17 channels at 64 equally spaced instants from the onset to the end
 * (inclusive), yaw_cum re-zeroed at the onset. FLOAT32 like the Python reference (it casts with
 * astype(np.float32)): store it in a Float32Array so every model reads the same rounded values.
 */
export interface ApproachWindow {
  /** (17, 64) channel-major: data[c * 64 + k] */
  readonly data: Float32Array;
  readonly onsetMs: number;
  readonly endMs: number;
}

// ───────────────────────────── 4. detect: bouts, segments, attempts ─────────────────────────────

/** m4_detect.SegParams (lane m4's production detector; the values come from the exported bundle). */
export interface SegParams {
  readonly lo_dps: number;
  readonly hi_dps: number;
  readonly sustain_ms: number;
  readonly end_sustain_ms: number;
  readonly smooth_hz: number;
  readonly causal: boolean;
  readonly min_duration_ms: number;
  readonly max_duration_s: number;
  readonly still_cap_s: number;
  readonly lookahead_s: number;
  readonly crop_s: number;
}

/** m4_detect.Segment: one hysteresis bout of the activity |w| (a candidate movement). */
export interface Bout {
  /** position in the stream's bout list (0-based) */
  readonly index: number;
  /** bounds moved back by the causal filter delay (ms on the stream clock) */
  readonly onsetMs: number;
  readonly endMs: number;
  readonly iOn: number;
  readonly iEnd: number;
  /** when a causal detector knows the bout: t[min(n - 1, iEnd + nLook - 1)] */
  readonly decidedMs: number;
  readonly stillBeforeS: number;
  readonly stillAfterS: number;
  readonly startedActive: boolean;
  /** false when the stream stopped before the stillness that ends the bout */
  readonly ended: boolean;
  readonly peakDps: number;
}

export type SegmentClass = 'approach' | 'return' | 'other';

/** One row of m4_detect.segment_table plus the segment classifier's output and the attempt rule. */
export interface SegmentRow {
  readonly bout: Bout;
  /** canonical() succeeded on the bout's bounds (false: no window, features NaN, never an attempt) */
  readonly canonicalOk: boolean;
  /** m4_detect.ALL_FEATURES by name (NaN when not computable) */
  readonly features: Readonly<Record<string, number>>;
  /** the (17, 64) window the hand models read; null when canonicalOk is false */
  readonly window: ApproachWindow | null;
  readonly closureDeg: number;
  readonly proba: Readonly<Record<SegmentClass, number>>;
  /** arg-max class */
  readonly cls: SegmentClass;
  /** FittedPipeline.attempts: cls 'approach' AND P(approach) >= det_threshold AND canonicalOk (no pairing) */
  readonly isAttempt: boolean;
}

/** m4_online.stopping_rule parameters (m4_config: p_stop 0.9, k_max 5, p_answer 0.5). */
export interface StoppingParams {
  readonly pStop: number;
  readonly kMax: number;
  readonly pAnswer: number;
}

/**
 * m4_online.stopping_rule's `stopped` values plus 'watching' (live: not decided yet) and 'grip'
 * (m4_production: the rest pose before the first movement has the screen facing away or down).
 */
export type DecisionStatus = 'watching' | 'confident' | 'k_max' | 'stream_end' | 'no_attempt' | 'grip';

/**
 * Port of m4_production.predict_stream's return value for one COMPLETE stream (the pure function the
 * parity tests compare with Python). NaN values of the reference become null here.
 */
export interface StreamPrediction {
  readonly modelId: ModelId;
  readonly pRight: number;
  /** int(round(100 * p)) with Python's round-half-even */
  readonly percentRight: number;
  readonly answered: boolean;
  readonly status: Exclude<DecisionStatus, 'watching'>;
  readonly nAttemptsUsed: number;
  readonly decidedMs: number | null;
  readonly restUpZ: number | null;
  readonly gripWarning: boolean;
  readonly attempts: readonly {
    readonly onsetMs: number;
    readonly endMs: number;
    readonly decidedMs: number;
    readonly pApproach: number;
    /** this attempt alone: sigmoid of its calibrated logit */
    readonly pRight: number;
  }[];
  readonly segments: readonly {
    readonly onsetMs: number;
    readonly endMs: number;
    readonly decidedMs: number;
    readonly pApproach: number;
    readonly pReturn: number;
    readonly pOther: number;
  }[];
}

// ───────────────────────────── 5. models ─────────────────────────────

export type ModelKind = 'cnn1d_ensemble' | 'logistic';

/**
 * How k attempts become one P(right) (each model has its own, fitted in Stage 2):
 *  - sum_over_tau (final CNN): P_k = sigmoid(sum of the k calibrated logits / tau_k), tau_1 = 1, k above the
 *    largest fitted k uses the largest (final_infer.aggregate);
 *  - scaled_mean (lane m4): P_k = sigmoid(s_k * mean of the k calibrated logits) (m4_online.aggregate
 *    'scaled'; s null = s_k 1, the plain mean logit);
 *  - mean / sum: sigmoid(mean) / sigmoid(sum) of the calibrated logits;
 *  - sum_over_k_pow (PHYS6, lane m1 fitted_alpha): P_k = sigmoid(sum of the k logits / k^alpha);
 *  - mean_p (yaw reference, the harness k-curve): P_k = mean of the k attempts' P (never fitted: not
 *    calibrated for k >= 2).
 */
export type AggregationSpec =
  | { readonly kind: 'sum_over_tau'; readonly tau: Readonly<Record<string, number>> }
  | { readonly kind: 'scaled_mean'; readonly s: Readonly<Record<string, number>> | null }
  | { readonly kind: 'mean' }
  | { readonly kind: 'sum' }
  | { readonly kind: 'sum_over_k_pow'; readonly alpha: number }
  | { readonly kind: 'mean_p' };

/**
 * A calibrator as exported (final_infer.calibrate and m4_models.CALIBRATORS). Maps the raw logit z to
 * P(right) of one attempt; p_clip (when set) clips p to [p_clip, 1 - p_clip].
 */
export type CalibratorJson =
  | { readonly method: 'none'; readonly logit_clip?: number; readonly p_clip?: number }
  | { readonly method: 'temperature'; readonly T: number; readonly logit_clip?: number; readonly p_clip?: number }
  | { readonly method: 'platt'; readonly a: number; readonly b: number; readonly logit_clip?: number; readonly p_clip?: number }
  | { readonly method: 'beta'; readonly a: number; readonly b: number; readonly c: number; readonly p0_clip?: number; readonly p_clip?: number }
  | {
      /** np.interp(z, x, y) */
      readonly method: 'isotonic';
      readonly x: readonly number[];
      readonly y: readonly number[];
      readonly p_clip?: number;
    }
  | {
      /**
       * lane m4's symmetric isotonic (m4_models), as exported: P = the fitted IsotonicRegression at |z| (the
       * thresholds, scipy interp1d, |z| clipped to [x_min, x_max]), clipped to [p_lo, p_hi], and 1 - P for z < 0
       */
      readonly method: 'isotonic_sym';
      readonly x_thresholds: readonly number[];
      readonly y_thresholds: readonly number[];
      readonly x_min: number;
      readonly x_max: number;
      readonly p_lo: number;
      readonly p_hi: number;
    }
  | { readonly method: 'stacked'; readonly inner: CalibratorJson; readonly outer: CalibratorJson; readonly p_clip?: number }
  | {
      /** lane m4's inner-CV choice among none, temperature and isotonic_sym (ModelSpec.calibration names the chosen one) */
      readonly method: 'nested';
      readonly chosen: string;
      readonly inner: CalibratorJson;
      readonly cv_log_loss?: Readonly<Record<string, number>>;
    };

/** What the UI and the registry know about a bundled model (models/registry.ts). */
export interface ModelSpec {
  readonly id: ModelId;
  /** the Stage 2 result name, e.g. 'm3_cal_deep_platt', 'm4_cal_nested', 'm1_lr_phys6_nested', 'example_yaw_logreg' */
  readonly stage2Name: string;
  /** selector label, e.g. 'Final CNN' (English; the UI translates by id) */
  readonly label: string;
  readonly kind: ModelKind;
  /** sha256 (hex) of the bundled JSON file's bytes; the footer shows the first 8 characters */
  readonly sha256: string;
  readonly calibration: CalibratorJson['method'];
  readonly aggregation: AggregationSpec;
  readonly trainedOn: { readonly persons: number; readonly trials: number };
  /** exactly one model is the default ('cnn') */
  readonly isDefault: boolean;
}

/** The inputs every hand model gets for one attempt (all models read the same window). */
export interface AttemptInput {
  readonly window: ApproachWindow;
  /**
   * The final model's screen_check input (final_robust.rest_up_z): the median of the canonical up_z channel
   * at the 30 fixed-window instants onset - 0.5 s + k / 60 s. When it is given and below
   * screen_check.threshold, the CNN un-flips the window first (final_infer.predict_proba(window, rest_up_z)).
   * The engine does NOT pass it (spec section 4.4: production asks the user to turn the phone instead), so
   * the app equals final_infer.predict_proba(window) without rest_up_z; only the un-flip parity test sets it.
   */
  readonly restUpZBefore?: number;
}

/** One model's evidence from one attempt. */
export interface ModelAttemptOutput {
  readonly modelId: ModelId;
  /** the model's raw logit (before its calibrator) */
  readonly zRaw: number;
  /** calibrated P(right) of this attempt alone */
  readonly pRight: number;
  /** calibrated logit (logit(pRight) with the model's own clipping): what the aggregation sums or averages */
  readonly z: number;
  /**
   * false when this model cannot score the attempt (PHYS6 with the screen exactly horizontal; a non-finite
   * window for the CNN, PHYS6 or the yaw reference): zRaw, pRight and z are then NaN, and the aggregation and
   * the stopping rule skip the attempt for this model only (PORT_SPEC 3.5). Absent = true.
   */
  readonly available?: boolean;
}

/** A loaded model (models/): pure, synchronous, no I/O. */
export interface HandModel {
  readonly spec: ModelSpec;
  predict(input: AttemptInput): ModelAttemptOutput;
  /** P_k of the first k = outputs.length attempts (k = 0 → 0.5) */
  aggregate(outputs: readonly ModelAttemptOutput[]): number;
}

/** final_model.json ('nosetouch-final-1'), as written by stage2/lanes/final_export.py (subset the app reads). */
export interface FinalModelJson {
  readonly format: 'nosetouch-final-1';
  readonly candidate: string;
  readonly label: string;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly input: {
    readonly n_samples: number;
    readonly channels: readonly CanonicalChannel[];
    readonly body_frame: { readonly channels: readonly string[]; readonly h_min: number };
    readonly preprocessing: Readonly<Record<string, number>>;
  };
  readonly model: {
    readonly kind: 'cnn1d_ensemble';
    readonly channels: readonly string[];
    readonly n_bins: number;
    readonly kernels: readonly number[];
    readonly width: number;
    readonly members: readonly {
      readonly center: readonly number[];
      readonly scale: readonly number[];
      readonly layers: readonly {
        /** (Cout, Cin, K) nested lists */
        readonly weight: readonly (readonly (readonly number[])[])[];
        readonly bn_weight: readonly number[];
        readonly bn_bias: readonly number[];
        readonly bn_mean: readonly number[];
        readonly bn_var: readonly number[];
        readonly bn_eps: number;
      }[];
      readonly fc_weight: readonly number[];
      readonly fc_bias: number;
    }[];
  };
  readonly calibrator: CalibratorJson;
  readonly aggregation: { readonly rule: string; readonly tau: Readonly<Record<string, number>> };
  readonly screen_check: { readonly rule: string; readonly threshold: number; readonly flip_signs: readonly number[] };
  readonly prior_right: number;
  readonly decision: { readonly threshold: number; readonly note: string };
  readonly training: { readonly n_trials: number; readonly n_persons: number };
}

// ───────────────────────────── 6. the live session engine (what the UI calls) ─────────────────────────────

/** One accepted attempt, frozen when the engine confirmed it (never recomputed afterwards). */
export interface AttemptResult {
  /** 1-based, in decision order */
  readonly k: number;
  readonly bout: Bout;
  readonly pApproach: number;
  readonly pReturn: number;
  readonly pOther: number;
  readonly closureDeg: number;
  /** QC / debug only: the rest pose before this movement (AttemptInput.restUpZBefore's definition); NaN if unknown */
  readonly restUpZBefore: number;
  /** every bundled model's output for this attempt (compare-all reads these) */
  readonly outputs: Readonly<Record<ModelId, ModelAttemptOutput>>;
  /** stream time (ms) at which the engine froze it: >= bout.decidedMs + settleMs */
  readonly frozenAtMs: number;
}

/** A confirmed movement that is not an attempt (a return, a fidget, a weak approach). Shown in debug only. */
export interface IgnoredMovement {
  readonly bout: Bout;
  readonly cls: SegmentClass;
  readonly pApproach: number;
  readonly reason: 'return' | 'other' | 'below_threshold' | 'no_window';
  readonly frozenAtMs: number;
}

/** The stopping rule's view of the session for one model (the driver's ends the session). */
export interface Decision {
  readonly modelId: ModelId;
  /** P_k of this model's aggregation over the attempts used (0.5 with none or under 'grip') */
  readonly pRight: number;
  /** int(round(100 * pRight)), Python round-half-even (the UI clamps the display to "<1 %" / ">99 %") */
  readonly percentRight: number;
  readonly nUsed: number;
  readonly status: DecisionStatus;
  /** the session is over (confident, k_max, stream_end, no_attempt at the time limit, grip) */
  readonly final: boolean;
  /** final AND an answer is shown (false = abstain: "Not sure - please repeat", or grip) */
  readonly answered: boolean;
  /** the more likely hand (pRight > 0.5 right, < 0.5 left, exactly 0.5 null) */
  readonly hand: Hand | null;
  /** stream time of the attempt that decided (null while watching or without attempts) */
  readonly decidedMs: number | null;
  /** P_1 .. P_nUsed of this model (the UI's small history) */
  readonly trajectory: readonly number[];
}

/** One row of the compare-all view: a model's P(right) after the SAME attempts the driver used. */
export interface ModelComparison {
  readonly modelId: ModelId;
  readonly pRight: number;
  readonly percentRight: number;
  readonly k: number;
  /** this model's own rule would call it: max(P, 1 - P) >= pStop */
  readonly confident: boolean;
}

/**
 * Engine phases. watching: buffering and confirming movements; final: the driver's stopping rule ended the
 * session (an answer or an abstention); grip: lane m4's grip gate failed (no answer until reset);
 * stopped: dispose() was called.
 */
export type EnginePhase = 'watching' | 'final' | 'grip' | 'stopped';

/** Short-lived events the UI may animate (a 30 ms vibration, a check mark). */
export type EngineEvent =
  | { readonly type: 'attempt'; readonly k: number; readonly atMs: number }
  | { readonly type: 'ignored'; readonly reason: IgnoredMovement['reason']; readonly atMs: number }
  | { readonly type: 'final'; readonly atMs: number }
  | { readonly type: 'reset'; readonly atMs: number }
  | { readonly type: 'gap'; readonly atMs: number };

/** Everything the UI renders, as one immutable value (a new object after every change). */
export interface EngineSnapshot {
  readonly phase: EnginePhase;
  /** stream time of the newest sample (ms since the stream started; 0 before the first sample) */
  readonly tMs: number;
  readonly nSamples: number;
  /** measured sample rate over the last 2 s (null before 2 samples) */
  readonly rateHz: number | null;
  /** the causal activity is below lo_dps now (UI: "hold still" turns into a calm dot) */
  readonly still: boolean;
  /**
   * UI hint only (never changes a result): the phone is still and the median up_z of the last 0.5 s is
   * below 0, i.e. the screen faces down or away. The answer-blocking check is the grip gate (phase 'grip').
   */
  readonly screenAwayHint: boolean;
  /** lane m4 rest_up_z of the current stream (null until the first bout is confirmed) */
  readonly restUpZ: number | null;
  readonly attempts: readonly AttemptResult[];
  readonly ignored: readonly IgnoredMovement[];
  /** the selected model, whose stopping rule ends the session */
  readonly driver: ModelId;
  readonly decision: Decision;
  /** every bundled model after the same attempts (registry order; includes the driver) */
  readonly compare: readonly ModelComparison[];
  readonly stopping: StoppingParams;
  readonly lastEvent: EngineEvent | null;
}

export interface SessionEngineOptions {
  readonly normalization: StreamNormalization;
  /** the bundled models (models/registry.ts loadModels()); the engine computes every one per attempt */
  readonly models: readonly HandModel[];
  readonly driver: ModelId;
  /** run the pipeline every tickMs of STREAM time (default 100) */
  readonly tickMs?: number;
  /**
   * freeze a bout only once the stream extends this far past its decision time (default 1000 ms):
   * the zero-phase low-pass then no longer changes any sample the bout depends on (spec section 4.2)
   */
  readonly settleMs?: number;
  /** without a final decision after this much stream time the session ends as 'stream_end' (default 180 s) */
  readonly maxStreamMs?: number;
  /**
   * run the models of a new attempt only once the phone is still again (the newest activity below lo_dps), or this
   * much stream time after its freeze (default 0: at once). The window is cut at the freeze tick either way, so the
   * numbers do not change; the heavy work (the CNN) just leaves the next movement. The page uses 1500.
   */
  readonly deferScoringMs?: number;
}

/**
 * The live loop (detect/engine.ts). Driven ONLY by sample times, never by wall-clock timers, so a replay of
 * the same samples gives the same snapshots (tests, the demo, the parity harness).
 */
export interface SessionEngine {
  /** append samples (source clock); runs a tick whenever the stream crossed the next tickMs boundary */
  push(samples: LiveSample | readonly LiveSample[]): void;
  /** choose the model whose stopping rule decides; recomputed at once from the frozen attempts */
  setDriver(id: ModelId): void;
  /** start a new stream with the next sample (Again / Start over); keeps the driver */
  reset(): void;
  /** a gap in the data (page hidden, sensor stall > 1 s): watching → reset; final stays final */
  notifyGap(): void;
  snapshot(): EngineSnapshot;
  /** called after every change; returns the unsubscribe function */
  subscribe(listener: (s: EngineSnapshot) => void): () => void;
  dispose(): void;
}

export type CreateSessionEngine = (options: SessionEngineOptions) => SessionEngine;

// ───────────────────────────── 7. seams between the owners ─────────────────────────────

/**
 * pipeline/liveBuffer.ts: the growing raw buffer of one stream. Rounds every sample exactly like
 * core/recorder.ts StreamRecorder.addMotionRel (roundValue + ROUNDING; t relative to the FIRST sample), so
 * raw() equals rawMotionFromRecord() of the record the collector would have stored for the same samples.
 */
export interface LiveBuffer {
  /** appends one sample; returns its index, or -1 when its t is not finite or not after the last t */
  push(s: LiveSample): number;
  readonly n: number;
  /** source-clock time of the first sample (stream t = 0); null when empty */
  readonly t0: number | null;
  /** a view of the first `n` samples (default: all); never copies more than it must */
  raw(n?: number): RawMotion;
  clear(): void;
}

/** pipeline/: the functions detect/ and the tests call (ports of the named Python functions). */
export interface PipelineApi {
  createLiveBuffer(): LiveBuffer;
  /** TrialRecord.streams.motion (null → NaN) → RawMotion (motionlab.io.streams_from_record) */
  rawMotionFromRecord(motion: {
    t: readonly number[];
    acc_x: readonly (number | null)[];
    acc_y: readonly (number | null)[];
    acc_z: readonly (number | null)[];
    accg_x: readonly (number | null)[];
    accg_y: readonly (number | null)[];
    accg_z: readonly (number | null)[];
    rr_alpha: readonly (number | null)[];
    rr_beta: readonly (number | null)[];
    rr_gamma: readonly (number | null)[];
  }): RawMotion;
  /** the production normalisation of an engine (webkit -1, every other engine +1; W3C deg/s) */
  productionNormalization(engine: BrowserEngine | null): StreamNormalization;
  /** standardize.stream_from_record steps 1-4 */
  streamFrom(raw: RawMotion, norm: StreamNormalization): Stream;
  /** m4_detect.crop_stream: the samples with startMs <= t <= stopMs */
  cropStream(stream: Stream, startMs: number, stopMs: number): Stream;
  /** standardize.canonical (raises RangeError where Python raises ValueError) */
  canonical(stream: Stream, onsetMs: number, endMs: number): Canonical;
  /** standardize.time_normalised (n = 64) */
  timeNormalised(can: Canonical, startMs: number, endMs: number): ApproachWindow;
}

/**
 * Lane m4's production detector as exported (models/json/m4_production.json, section 'detector'): read by
 * detect/, written by the Python exporter from ProductionModel.describe() and the fitted SegmentClassifier.
 * PROPOSED shape: the exporter and detect/ agree on it through this interface.
 */
export interface DetectorBundle {
  readonly segParams: SegParams;
  /** m4_detect.causal_delay_ms(segParams): subtracted from causal bounds */
  readonly causalDelayMs: number;
  /** scipy butter(2, smooth_hz, fs=60, output='sos') and sosfilt_zi of it (causal activity filter) */
  readonly activitySos: readonly (readonly number[])[];
  readonly activityZi: readonly (readonly number[])[];
  readonly segmentClassifier: {
    /** m4_detect feature names in column order (the 'robust' set) */
    readonly features: readonly string[];
    /** sklearn classes_ order */
    readonly classes: readonly SegmentClass[];
    readonly scaling: 'robust' | 'z' | 'none';
    /** (F - center) / scale, NaN → 0, clipped to +-clip (none: NaN → fill, no clip) */
    readonly center: readonly number[];
    readonly scale: readonly number[];
    readonly fill: readonly number[] | null;
    readonly clip: number | null;
    /** multinomial logistic: softmax(coef @ z + intercept); coef (n_classes, n_features) */
    readonly coef: readonly (readonly number[])[];
    readonly intercept: readonly number[];
  };
  /** FittedPipeline.attempts: P(approach) >= detThreshold; pairReturns false in production */
  readonly attemptRule: { readonly detThreshold: number; readonly pairReturns: boolean };
  readonly stopping: StoppingParams;
  /**
   * m4_config GRIP_GATE_UPZ (null = off) and GRIP_REST_MIN_S. rest_up_z runs with m4_detect.DEFAULT
   * (ProductionModel.describe()['grip_rest_params']), which may differ from segParams (the fitted SEG_GRID entry).
   */
  readonly grip: {
    readonly gateUpz: number | null;
    readonly restMinS: number;
    readonly restParams: SegParams;
    /** causal_delay_ms(restParams) (exported; equal to causalDelayMs while both detectors use the same 5 Hz filter) */
    readonly restCausalDelayMs?: number;
  };
  /** stage2 config constants the pipeline needs (FS_HZ, LOWPASS_HZ, MAX_GAP_MS, STILL_GYRO_DPS, REST_MS, N_NORM) */
  readonly constants: Readonly<Record<string, number>>;
}

/** detect/: the pure functions plus the live engine. */
export interface DetectApi {
  /** m4_detect.activity (causal) on a stream */
  activity(stream: Stream, p: SegParams): Float64Array;
  /** m4_detect.find_segments */
  findBouts(stream: Stream, p: SegParams, activity?: Float64Array): Bout[];
  /** m4_detect.segment_table + the segment classifier + the attempt rule */
  segmentRows(stream: Stream, detector: DetectorBundle): SegmentRow[];
  /** m4_detect.rest_up_z (NaN → null) */
  restUpZ(stream: Stream, detector: DetectorBundle): number | null;
  /** m4_online.stopping_rule over attempts in decision order (streamEnded false → 'watching' instead of 'stream_end') */
  stoppingRule(model: HandModel, attempts: readonly AttemptResult[], params: StoppingParams, streamEnded: boolean): Decision;
  /** m4_production.predict_stream with `model` as the hand classifier (the m4 model reproduces Python exactly) */
  predictStream(raw: RawMotion, norm: StreamNormalization, model: HandModel, detector: DetectorBundle): StreamPrediction;
  createSessionEngine: CreateSessionEngine;
}

/** models/registry.ts */
export interface ModelsApi {
  /** every bundled model, registry order: cnn, m4, phys6, yaw */
  readonly specs: readonly ModelSpec[];
  loadModels(): HandModel[];
  /** lane m4's detector (same JSON file as the 'm4' model) */
  detector(): DetectorBundle;
  readonly defaultModel: ModelId;
}

// ───────────────────────────── 8. UI screens (data-testid values the e2e tests use) ─────────────────────────────

/**
 * The screen shown, derived from the preflight, the source and the engine snapshot (PredictApp.tsx).
 * Each screen's root element carries data-testid = `pm-${screen}`.
 */
export type PredictScreen =
  | 'start' // one big Start button
  | 'permission' // waiting for the iOS / Chrome permission prompt
  | 'watching' // sensing; no attempt yet, or attempts but no final decision (provisional P shown)
  | 'answer' // final and answered: P(right) = 87 %
  | 'abstain' // final, not answered: "Not sure - please repeat"
  | 'grip' // the grip gate failed: "Turn the screen towards your face"
  | 'unsupported'; // a SourceFailure (one message per reason)
