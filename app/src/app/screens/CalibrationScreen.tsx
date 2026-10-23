/**
 * Calibration (optional, ~30 s): measures instead of assuming the platform's sensor conventions.
 *  1. phone flat on a table, screen up, untouched → poseStats → gravityFromFlat (gravity sign/size)
 *  2. phone upright facing the participant → uprightCheck
 *  3. tilt test (flat → tilt right onto its side → back) → axisCheck (rotationRate mapping/units)
 * Each step: countdown (hands off / get into position; also lets the sensor pipeline settle), then a
 * short recording window. buildCalibration() → state.calibration when the participant continues.
 * The math lives in core/calibration.ts; this screen only records the windows and explains results.
 * A flat sign that contradicts the browser engine (state.device.engine: webkit −1, blink/gecko +1)
 * is shown as a warning (screen-down phone?), never as success; continuing anyway keeps it with the
 * note FLAT_SIGN_CONTRADICTS_ENGINE. An 'inverted' upright check asks to repeat steps 1 and 2.
 * Simulation mode has no engine expectation (the simulator's sign is not the browser's).
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import {
  axisCheck,
  buildCalibration,
  engineGravitySign,
  gravityFromFlat,
  poseStats,
  uprightCheck,
} from '../../core/calibration';
import type {
  AxisCheck,
  CalibrationResult,
  CueKind,
  MotionSample,
  PoseStats,
  SensorError,
  SensorListener,
  SensorSource,
  TrialCondition,
} from '../../core/types';
import { isoTime } from '../../core/util';
import { useStrings, type Strings } from '../../i18n';
import { Button, Card, Notice, ProgressBar, Screen } from '../../ui/components';
import '../../ui/onboarding.css';
import { DEFAULT_CONDITION, getServices } from '../services';
import { navigate, setState, showToast, useAppState } from '../store';

export type CalibrationStep = 'flat' | 'upright' | 'tilt';

export const STEP_TIMING: Record<CalibrationStep, { leadMs: number; durationMs: number }> = {
  flat: { leadMs: 3000, durationMs: 2000 },
  upright: { leadMs: 2000, durationMs: 2000 },
  tilt: { leadMs: 2000, durationMs: 3500 },
};

/** Fewer motion samples than this in a window means the sensors did not deliver (2 s at 60 Hz ≈ 120). */
export const MIN_WINDOW_SAMPLES = 10;

export interface RecordedWindow {
  samples: MotionSample[];
  error: SensorError | null;
}

/**
 * Starts `source`, ignores the first `leadMs`, keeps the motion samples of the following
 * `durationMs` (source clock), then stops the source. Never rejects; `signal` ends it early with what
 * was collected so far (the screen was left).
 */
export function recordWindow(
  source: SensorSource,
  leadMs: number,
  durationMs: number,
  signal?: AbortSignal,
): Promise<RecordedWindow> {
  return new Promise((resolve) => {
    const samples: MotionSample[] = [];
    let error: SensorError | null = null;
    let from = Number.POSITIVE_INFINITY;
    let to = Number.POSITIVE_INFINITY;
    let settled = false;
    let leadTimer: ReturnType<typeof setTimeout> | undefined;
    let endTimer: ReturnType<typeof setTimeout> | undefined;

    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(leadTimer);
      clearTimeout(endTimer);
      signal?.removeEventListener('abort', finish);
      try {
        source.stop();
      } catch {
        // already stopped
      }
      resolve({ samples, error });
    };

    if (signal?.aborted) {
      resolve({ samples, error });
      return;
    }
    signal?.addEventListener('abort', finish);

    const listener: SensorListener = {
      onMotion: (sample) => {
        if (!settled && sample.t >= from && sample.t <= to) samples.push(sample);
      },
      onOrientation: () => undefined,
      onError: (e) => {
        error = e;
      },
    };

    leadTimer = setTimeout(() => {
      from = source.clock.now();
      to = from + durationMs;
      // Scheduled from the actual window start so a late timer cannot shorten the window.
      endTimer = setTimeout(finish, durationMs + 30);
    }, leadMs);

    const failed = (e: unknown) => {
      error = { code: 'sensor_error', message: e instanceof Error ? e.message : String(e) };
      finish();
    };
    try {
      source.start(listener).catch(failed);
    } catch (e) {
      failed(e);
    }
  });
}

/** true when every number inside `v` is finite (NaN/Infinity would become null in JSON and break the schema). */
export function allFinite(v: unknown): boolean {
  if (typeof v === 'number') return Number.isFinite(v);
  if (Array.isArray(v)) return v.every(allFinite);
  if (v && typeof v === 'object') return Object.values(v).every(allFinite);
  return true;
}

/** 'face_down': a sign the engine expectation contradicts (|z| ≈ |g| also when lying screen-down). */
export type FlatVerdict = 'ok' | 'ok_inverted' | 'face_down' | 'moving' | 'not_flat' | 'odd';
export type UprightVerdict = 'ok' | 'inverted' | 'moving' | 'unclear';
export type AxisVerdict = 'ok' | 'noted' | 'radians' | 'unclear';

/**
 * Why a flat measurement did (not) give a usable gravity sign — mirrors the gravityFromFlat rules.
 * `expected`: engineGravitySign(device.engine); null (unknown engine, simulation) accepts either sign.
 */
export function flatVerdict(stats: PoseStats | null, sign: 1 | -1 | null, expected: 1 | -1 | null = null): FlatVerdict | null {
  if (!stats) return null;
  if (sign !== null && expected !== null && sign !== expected) return 'face_down';
  if (sign === 1) return 'ok';
  if (sign === -1) return 'ok_inverted';
  if (!stats.still) return 'moving';
  const [x, y, z] = stats.accg_mean;
  const g = Math.hypot(x, y, z);
  if (!(g >= 8.6 && g <= 11)) return 'odd';
  return Math.abs(z) < 0.9 * g ? 'not_flat' : 'odd';
}

export function uprightVerdict(stats: PoseStats | null, check: CalibrationResult['upright_check']): UprightVerdict | null {
  if (!stats) return null;
  if (check === 'ok') return 'ok';
  if (check === 'inverted') return 'inverted';
  return stats.still ? 'unclear' : 'moving';
}

export function axisVerdict(a: AxisCheck | null): AxisVerdict | null {
  if (!a) return null;
  if (a.verdict === 'unclear') return 'unclear';
  if (a.units === 'rad_per_s') return 'radians';
  return a.verdict === 'w3c' ? 'ok' : 'noted';
}

/** [text, notice kind] of a step result. Only 'ok' marks the step done (✓); 'warn' asks to measure again. */
export type ResultMessage = [string, 'ok' | 'warn'];

/**
 * Text and kind of every verdict. A screen-down flat step ('face_down') and an 'inverted' upright
 * check (steps 1 and 2 disagree) are warnings asking to measure again, never a success.
 */
export function resultMessages(t: Strings['onboarding']['calibration']): {
  flat: Record<FlatVerdict, ResultMessage>;
  upright: Record<UprightVerdict, ResultMessage>;
  axis: Record<AxisVerdict, ResultMessage>;
} {
  return {
    flat: {
      ok: [t.flatOk, 'ok'],
      ok_inverted: [t.flatOkInverted, 'ok'],
      face_down: [t.flatFaceDown, 'warn'],
      moving: [t.flatMoving, 'warn'],
      not_flat: [t.flatNotFlat, 'warn'],
      odd: [t.flatOdd, 'warn'],
    },
    upright: {
      ok: [t.uprightOk, 'ok'],
      inverted: [t.uprightInverted, 'warn'],
      moving: [t.uprightMoving, 'warn'],
      unclear: [t.uprightUnclear, 'warn'],
    },
    axis: {
      ok: [t.axisOk, 'ok'],
      noted: [t.axisNoted, 'ok'],
      radians: [t.axisRadians, 'ok'],
      unclear: [t.axisUnclear, 'warn'],
    },
  };
}

function safePlay(kind: CueKind): void {
  try {
    getServices().cues.play(kind);
  } catch {
    // cues are a convenience
  }
}

function safeGravity(flat: PoseStats | null): 1 | -1 | null {
  try {
    return gravityFromFlat(flat).sign;
  } catch (e) {
    console.warn('gravityFromFlat failed', e);
    return null;
  }
}

function safeUpright(upright: PoseStats | null, sign: 1 | -1 | null): CalibrationResult['upright_check'] {
  try {
    return uprightCheck(upright, sign);
  } catch (e) {
    console.warn('uprightCheck failed', e);
    return null;
  }
}

interface ActiveStep {
  step: CalibrationStep;
  startedAt: number;
  leadMs: number;
  durationMs: number;
}

export function CalibrationScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.calibration;
  const simulate = st.settings.simulate;
  const [earlier] = useState<CalibrationResult | null>(st.calibration);
  const [flat, setFlat] = useState<PoseStats | null>(earlier?.flat ?? null);
  const [upright, setUpright] = useState<PoseStats | null>(earlier?.upright ?? null);
  const [axis, setAxis] = useState<AxisCheck | null>(earlier?.axis_check ?? null);
  const [measuredAt, setMeasuredAt] = useState<string | null>(earlier?.performed_at ?? null);
  const [noData, setNoData] = useState<Partial<Record<CalibrationStep, boolean>>>({});
  const [active, setActive] = useState<ActiveStep | null>(null);
  const [now, setNow] = useState(0);
  const activeRef = useRef<ActiveStep | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Leaving the screen stops a recording that is still running.
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(id);
  }, [active]);

  const store = (step: CalibrationStep, samples: MotionSample[]) => {
    const tooFew = samples.length < MIN_WINDOW_SAMPLES;
    setNoData((m) => ({ ...m, [step]: tooFew }));
    try {
      if (step === 'tilt') {
        const a = tooFew ? null : axisCheck(samples);
        setAxis(a && allFinite(a) ? a : null);
      } else {
        const p = tooFew ? null : poseStats(samples);
        const ok = p && allFinite(p) ? p : null;
        if (step === 'flat') setFlat(ok);
        else setUpright(ok);
      }
    } catch (e) {
      console.error(`calibration step ${step} failed`, e);
      showToast(s.common.errorGeneric, 'error');
    }
  };

  const runStep = async (step: CalibrationStep) => {
    if (activeRef.current) return;
    let source: SensorSource;
    try {
      const services = getServices();
      void services.cues.unlock(); // inside the tap, so the beeps below can play
      const condition: TrialCondition = {
        ...DEFAULT_CONDITION,
        orientation: step === 'upright' ? 'upright_toward' : 'flat_up',
      };
      source = services.createSource({ condition, behavior: step === 'flat' ? 'static' : 'no_move' });
    } catch (e) {
      console.error('no sensor source', e);
      showToast(s.common.errorGeneric, 'error');
      return;
    }
    const { leadMs, durationMs } = STEP_TIMING[step];
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const a: ActiveStep = { step, startedAt: Date.now(), leadMs, durationMs };
    activeRef.current = a;
    setActive(a);
    setNow(a.startedAt);
    // Beep just before the window opens (the "start tilting" signal of step 3). Never vibrate here.
    const beep = setTimeout(() => safePlay('ready'), Math.max(0, leadMs - 250));
    const { samples } = await recordWindow(source, leadMs, durationMs, ctrl.signal);
    clearTimeout(beep);
    activeRef.current = null;
    if (ctrl.signal.aborted) return; // screen left or calibration skipped: discard
    setActive(null);
    safePlay('end');
    setMeasuredAt(isoTime());
    store(step, samples);
  };

  // The simulator plays an Android-like phone whatever the browser: no engine expectation then.
  const engine = simulate ? null : (st.device?.engine ?? null);
  const sign = safeGravity(flat);
  const flatV = flatVerdict(flat, sign, engineGravitySign(engine));
  const uprightV = uprightVerdict(upright, upright ? safeUpright(upright, sign) : null);
  const axisV = axisVerdict(axis);
  const anyResult = flat !== null || upright !== null || axis !== null;

  const onContinue = () => {
    abortRef.current?.abort();
    let calibration: CalibrationResult | null = null;
    if (anyResult) {
      try {
        // A contradicted sign is kept: buildCalibration notes it (FLAT_SIGN_CONTRADICTS_ENGINE).
        calibration = buildCalibration({
          performedAt: measuredAt ?? isoTime(),
          flat,
          upright,
          axis,
          notes: simulate ? ['simulated: tilt step not performed by the simulator'] : [],
          engine,
        });
      } catch (e) {
        console.error('buildCalibration failed', e);
      }
    }
    setState({ calibration });
    navigate('setup');
  };

  const onSkip = () => {
    abortRef.current?.abort();
    activeRef.current = null;
    setState({ calibration: null });
    navigate('setup');
  };

  const msgs = resultMessages(t);

  const live = (step: CalibrationStep) => {
    if (!active || active.step !== step) return null;
    const elapsed = Math.max(0, now - active.startedAt);
    if (elapsed < active.leadMs) {
      return (
        <div class="onb-countdown" role="status" aria-live="polite">
          {t.getReady(Math.max(1, Math.ceil((active.leadMs - elapsed) / 1000)))}
        </div>
      );
    }
    return (
      <ProgressBar value={(elapsed - active.leadMs) / active.durationMs} label={step === 'tilt' ? t.tiltNow : t.holdStill} />
    );
  };

  const result = (step: CalibrationStep, msg: ResultMessage | null) => {
    if (active?.step === step) return null;
    if (noData[step]) return <Notice kind="error" testId={`calib-${step}-result`}>{t.noData}</Notice>;
    if (!msg) return null;
    return (
      <Notice kind={msg[1]} testId={`calib-${step}-result`}>
        {msg[0]}
      </Notice>
    );
  };

  const stepCard = (n: number, step: CalibrationStep, title: string, how: string, msg: ResultMessage | null, extra?: string) => {
    // ✓ only for an 'ok' result; a warning asks for another measurement.
    const done = !noData[step] && msg !== null && msg[1] === 'ok';
    return (
      <Card testId={`calib-step-${step}`}>
        <div class="stack-sm">
          <div class="onb-step-head">
            <span class={`onb-step-num ${done ? 'is-done' : ''}`} aria-hidden="true">
              {done ? '✓' : n}
            </span>
            <div>
              <div class="small muted">{t.stepLabel(n)}</div>
              <h2>{title}</h2>
            </div>
          </div>
          <p>{how}</p>
          {extra && <Notice kind="info">{extra}</Notice>}
          {live(step)}
          {result(step, msg)}
          <Button
            testId={`btn-calibrate-${step}`}
            variant={done ? 'secondary' : 'primary'}
            block
            disabled={active !== null}
            onClick={() => void runStep(step)}
          >
            {done || noData[step] || msg !== null ? t.measureAgain : t.measure}
          </Button>
        </div>
      </Card>
    );
  };

  return (
    <Screen
      title={t.title}
      onBack={() => navigate('sensors')}
      testId="screen-calibration"
      footer={
        <>
          <Button testId="btn-continue" size="lg" block disabled={active !== null || !anyResult} onClick={onContinue}>
            {t.continue}
          </Button>
          <Button testId="btn-skip-calibration" variant="ghost" block onClick={onSkip}>
            {t.skip}
          </Button>
        </>
      }
    >
      <p class="onb-lead">{t.intro}</p>
      {earlier && <Notice kind="info">{t.doneEarlier}</Notice>}
      {stepCard(1, 'flat', t.flatTitle, t.flatHow, flatV ? msgs.flat[flatV] : null)}
      {stepCard(2, 'upright', t.uprightTitle, t.uprightHow, uprightV ? msgs.upright[uprightV] : null)}
      {stepCard(3, 'tilt', t.tiltTitle, t.tiltHow, axisV ? msgs.axis[axisV] : null, simulate ? t.tiltSim : undefined)}
    </Screen>
  );
}
