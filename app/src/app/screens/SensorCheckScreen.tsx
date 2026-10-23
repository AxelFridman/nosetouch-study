/**
 * Sensor check: explains why motion access is needed, requests it inside the tap (WebKit only shows
 * the prompt while a user gesture is being processed), measures the real sampling rates for 2.5 s and
 * stores permission + measurement in state.capabilities (snapshotted into every session record).
 * Gives platform-specific help when access is denied, no events arrive or the rate is too low.
 */
import { useEffect, useState } from 'preact/hooks';
import { detectCapabilities, measureRates, requestMotionPermission } from '../../core/sensors';
import type { DeviceInfo, MeasuredRates, PermissionState, SensorCapabilities } from '../../core/types';
import { useStrings, type Strings } from '../../i18n';
import { Button, Card, Notice, ProgressBar, Screen, Stat } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { getState, navigate, setState, useAppState } from '../store';

/** Length of the rate measurement. */
export const MEASURE_MS = 2500;
/** Below this the movement cannot be recorded usefully: continuing is blocked. */
export const MIN_MOTION_HZ = 20;
/** Typical browsers deliver ~60 Hz; below this we warn (Low Power Mode, throttling). */
export const GOOD_MOTION_HZ = 45;

/**
 * Starts the motion-permission request. Call it as the FIRST statement of a tap handler, before any
 * await: WebKit only shows the prompt while the gesture is being processed. Never throws/rejects
 * ('unknown' when the request itself failed).
 */
export function safeRequestPermission(): Promise<PermissionState> {
  try {
    return requestMotionPermission().catch((e: unknown) => {
      console.warn('motion permission request failed', e);
      return 'unknown' as const;
    });
  } catch (e) {
    console.warn('motion permission request failed', e);
    return Promise.resolve('unknown');
  }
}

/** Capabilities when feature detection itself failed: conservative, but schema-valid. */
export function fallbackCapabilities(): SensorCapabilities {
  return {
    secure_context: typeof isSecureContext === 'boolean' ? isSecureContext : false,
    device_motion: false,
    device_orientation: false,
    permission_api: false,
    permission_state: 'unknown',
    generic_relative_orientation: false,
    wake_lock: false,
    vibrate: false,
    share_files: false,
    measured: null,
  };
}

export function currentCapabilities(): SensorCapabilities {
  const caps = getState().capabilities;
  if (caps) return caps;
  try {
    return detectCapabilities();
  } catch (e) {
    console.warn('detectCapabilities failed', e);
    return fallbackCapabilities();
  }
}

export function patchCapabilities(patch: Partial<SensorCapabilities>): void {
  setState({ capabilities: { ...currentCapabilities(), ...patch } });
}

export type SensorIssue = 'insecure' | 'denied' | 'failed' | 'no_events' | 'too_slow' | 'slow' | 'no_gyro';

export interface SensorVerdict {
  canContinue: boolean;
  issues: SensorIssue[];
  /** rounded motion rate, or null when nothing was measured */
  hz: number | null;
}

/** What the check found, independent of the UI (unit-tested). */
export function sensorVerdict(input: {
  simulate: boolean;
  secureContext: boolean;
  permission: PermissionState | null;
  measured: MeasuredRates | null;
  failed: boolean;
}): SensorVerdict {
  const raw = input.measured?.motion_hz ?? null;
  const hz = raw !== null && Number.isFinite(raw) ? Math.round(raw) : null;
  // The simulator delivers data regardless of the browser, so nothing can block the flow.
  if (input.simulate) return { canContinue: true, issues: [], hz };
  const issues: SensorIssue[] = [];
  if (!input.secureContext) issues.push('insecure');
  if (input.permission === 'denied') issues.push('denied');
  if (input.failed) issues.push('failed');
  if (input.measured) {
    if (hz === null || hz <= 0) issues.push('no_events');
    else if (hz < MIN_MOTION_HZ) issues.push('too_slow');
    else if (hz < GOOD_MOTION_HZ) issues.push('slow');
    if (hz !== null && hz > 0 && !input.measured.rotation_available) issues.push('no_gyro');
  }
  const canContinue = input.permission !== 'denied' && raw !== null && raw >= MIN_MOTION_HZ;
  return { canContinue, issues, hz };
}

function isIos(device: DeviceInfo | null): boolean {
  return device?.platform === 'ios' || device?.engine === 'webkit';
}

/**
 * Where a denied motion permission has to be reset. WebKit keeps the answer in memory per app
 * process and data store (deleting website data is no reliable reset), so on iOS the advice
 * depends on which app runs the page: Safari, the Home Screen web app, another browser (Chrome,
 * Firefox, Edge, ... for iOS) or the browser inside another app. Never a Private tab: that is a new,
 * empty storage (a new participant code, no resumable session, unsent data left behind).
 */
export type DeniedContext =
  | { kind: 'ios_safari' }
  | { kind: 'ios_home_screen' }
  | { kind: 'ios_browser'; browser: string }
  | { kind: 'ios_in_app' }
  | { kind: 'android' }
  | { kind: 'other' };

/** Name of a non-Safari iOS browser from DeviceInfo.browser ('Chrome iOS 141.0' → 'Chrome'), else null. */
export function iosBrowserName(browser: string | null | undefined): string | null {
  const name = (browser ?? '')
    .trim()
    .replace(/\s+\d[\d.]*$/, '')
    .replace(/\s+iOS$/, '')
    .trim();
  return name && !/^(?:Safari|iOS WebView|unknown)$/i.test(name) ? name : null;
}

export function deniedContext(device: DeviceInfo | null): DeniedContext {
  if (!isIos(device)) return { kind: device?.platform === 'android' ? 'android' : 'other' };
  if (device?.in_app_browser) return { kind: 'ios_in_app' };
  if (device?.standalone) return { kind: 'ios_home_screen' };
  const browser = iosBrowserName(device?.browser);
  return browser ? { kind: 'ios_browser', browser } : { kind: 'ios_safari' };
}

/** The sensor check's advice after a denial, for the browser context the page runs in. */
export function deniedText(t: Strings['onboarding']['sensors'], device: DeviceInfo | null): string {
  const ctx = deniedContext(device);
  switch (ctx.kind) {
    case 'ios_safari':
      return t.deniedIos;
    case 'ios_home_screen':
      return t.deniedIosHomeScreen;
    case 'ios_browser':
      return t.deniedIosBrowser(ctx.browser);
    case 'ios_in_app':
      return t.deniedIosInApp;
    case 'android':
      return t.deniedAndroid;
    default:
      return t.deniedOther;
  }
}

export function issueText(
  issue: SensorIssue,
  t: Strings['onboarding']['sensors'],
  device: DeviceInfo | null,
  hz: number | null,
): string {
  switch (issue) {
    case 'insecure':
      return t.insecure;
    case 'denied':
      return deniedText(t, device);
    case 'failed':
      return t.failed;
    case 'no_events':
      return device?.platform === 'other' ? `${t.noEventsDesktop} ${t.noEvents}` : t.noEvents;
    case 'too_slow':
      return t.tooSlow(hz ?? 0);
    case 'slow':
      return t.slow(hz ?? 0);
    default:
      return t.noGyro;
  }
}

const ISSUE_TONE: Record<SensorIssue, 'error' | 'warn'> = {
  insecure: 'error',
  denied: 'error',
  failed: 'error',
  no_events: 'error',
  too_slow: 'error',
  slow: 'warn',
  no_gyro: 'warn',
};

type Phase = 'idle' | 'requesting' | 'measuring' | 'done';

export function SensorCheckScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.sensors;
  const caps = st.capabilities;
  const measured = caps?.measured ?? null;
  const [phase, setPhase] = useState<Phase>(measured ? 'done' : 'idle');
  // Permission asked on THIS screen; earlier results (e.g. coming back from calibration) count too.
  const [permission, setPermission] = useState<PermissionState | null>(measured ? (caps?.permission_state ?? null) : null);
  const [failed, setFailed] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (phase !== 'measuring') return;
    const started = Date.now();
    setProgress(0);
    const id = setInterval(() => setProgress(Math.min(1, (Date.now() - started) / MEASURE_MS)), 100);
    return () => clearInterval(id);
  }, [phase]);

  const onRequest = async () => {
    // FIRST statement, before any await (WebKit gesture rule). Then unlock audio inside the same tap.
    const pending = safeRequestPermission();
    try {
      void getServices().cues.unlock();
    } catch (e) {
      console.warn('audio unlock failed', e);
    }
    if (phase === 'requesting' || phase === 'measuring') return;
    setFailed(false);
    setPhase('requesting');
    let perm = await pending;
    if (getState().settings.simulate) perm = 'granted';
    setPermission(perm);
    patchCapabilities({ permission_state: perm, measured: null });
    if (perm === 'denied') {
      setPhase('done');
      return;
    }
    setPhase('measuring');
    try {
      const rates = await measureRates(getServices().createSource(), MEASURE_MS);
      patchCapabilities({ permission_state: perm, measured: rates });
    } catch (e) {
      console.error('sensor measurement failed', e);
      setFailed(true);
    }
    setPhase('done');
  };

  const onBeep = () => {
    try {
      const cues = getServices().cues;
      void cues
        .unlock()
        .then(() => cues.play('go'))
        .catch(() => undefined);
    } catch (e) {
      console.warn('test beep failed', e);
    }
  };

  const verdict = sensorVerdict({
    simulate: st.settings.simulate,
    secureContext: caps?.secure_context ?? false,
    permission,
    measured,
    failed,
  });
  const busy = phase === 'requesting' || phase === 'measuring';
  const device = st.device;
  const fmtHz = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? t.none : t.hz(Math.round(v)));
  const avail = (ok: boolean) => (ok ? <span class="onb-yes">{t.available}</span> : <span class="onb-no">{t.missing}</span>);

  return (
    <Screen
      title={t.title}
      onBack={() => navigate('profile')}
      testId="screen-sensorcheck"
      footer={
        <Button testId="btn-continue" size="lg" block disabled={busy || !verdict.canContinue} onClick={() => navigate('calibration')}>
          {s.common.continue}
        </Button>
      }
    >
      <p class="onb-lead">{t.intro}</p>
      <Card title={t.stepsTitle}>
        <ol class="onb-steps">
          {t.steps.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ol>
        <p class="small muted">{t.privacyNote}</p>
      </Card>

      <Button
        testId="btn-request-permission"
        size="lg"
        block
        variant={phase === 'done' ? 'secondary' : 'primary'}
        disabled={busy}
        onClick={() => void onRequest()}
      >
        {phase === 'done' ? t.measureAgain : t.allow}
      </Button>

      {phase === 'requesting' && <Notice kind="info">{t.requesting}</Notice>}
      {phase === 'measuring' && <ProgressBar value={progress} label={t.measuring} />}

      {phase === 'done' && (
        <>
          {verdict.issues.map((issue) => (
            <Notice key={issue} kind={ISSUE_TONE[issue]} testId={`sensor-issue-${issue}`}>
              {issueText(issue, t, device, verdict.hz)}
            </Notice>
          ))}
          {verdict.canContinue && verdict.issues.length === 0 && (
            <Notice kind="ok" testId="sensor-ok">
              {st.settings.simulate ? t.simOk : t.ok}
            </Notice>
          )}
        </>
      )}

      {(phase === 'done' || measured) && (
        <Card title={t.resultsTitle} testId="sensor-results">
          <Stat label={t.motionRate} value={<span data-testid="sensor-rate">{fmtHz(measured?.motion_hz)}</span>} />
          <Stat label={t.orientationRate} value={fmtHz(measured?.orientation_hz)} />
          <Stat label={t.gyroscope} value={avail(!!measured?.rotation_available)} />
          <Stat label={t.linearAcc} value={avail(!!measured?.acc_available)} />
          <Stat label={t.quaternion} value={avail((measured?.quaternion_hz ?? 0) > 0)} />
          {device && (
            <Stat
              label={t.browser}
              value={
                <bdi>
                  {device.browser}
                  {device.engine && device.engine !== 'unknown' ? ` · ${device.engine}` : ''}
                </bdi>
              }
            />
          )}
        </Card>
      )}

      <Card title={t.tipsTitle}>
        <ul class="onb-checklist">
          {t.tips.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
        <Button testId="btn-test-beep" variant="secondary" block onClick={onBeep}>
          {t.testBeep}
        </Button>
      </Card>
    </Screen>
  );
}
