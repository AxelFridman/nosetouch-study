/**
 * Sensor check, v2: the REPAIR screen. It is not on the participant path: the welcome screen's Start
 * tap already requested the motion permission and measured the rates in the background
 * (app/sensorCheck.ts). This screen opens only when that check found that the sensors cannot record
 * (permission denied, no events, below MIN_MOTION_HZ, no secure context), or when resuming a session
 * needs the permission again. It explains what to do for the browser context the page runs in, lets
 * the participant allow access and measure again (the request is the FIRST statement of the tap:
 * WebKit only shows the prompt while the gesture is being processed), and Continue then goes on to
 * the session (or back to the welcome screen's resume card).
 */
import { useEffect, useState } from 'preact/hooks';
import type { DeviceInfo, PermissionState } from '../../core/types';
import { useStrings, type Strings } from '../../i18n';
import { Button, Card, Notice, ProgressBar, Screen, Stat } from '../../ui/components';
import '../../ui/onboarding.css';
import {
  beginSensorCheck,
  getRepairNext,
  lastSensorCheck,
  MEASURE_MS,
  safeRequestPermission,
  sensorVerdict,
  type SensorIssue,
} from '../sensorCheck';
import { getServices } from '../services';
import { continueToSession, unlockInTap } from '../startFlow';
import { createTapGuard } from '../state';
import { navigate, showToast, useAppState } from '../store';

// The permission and measurement helpers live in app/sensorCheck.ts; re-exported for older imports.
export {
  currentCapabilities,
  fallbackCapabilities,
  GOOD_MOTION_HZ,
  MEASURE_MS,
  MIN_MOTION_HZ,
  patchCapabilities,
  safeRequestPermission,
  sensorVerdict,
  type SensorIssue,
  type SensorVerdict,
} from '../sensorCheck';

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
  // What the background check (or an earlier visit of this screen) found is shown at once; a denial
  // seen since (e.g. by "Resume session") wins over an older result.
  const [earlier] = useState(lastSensorCheck);
  const deniedNow = caps?.permission_state === 'denied';
  const [phase, setPhase] = useState<Phase>(earlier !== null || measured !== null || deniedNow ? 'done' : 'idle');
  const [permission, setPermission] = useState<PermissionState | null>(
    deniedNow ? 'denied' : (earlier?.permission ?? (measured ? (caps?.permission_state ?? null) : null)),
  );
  const [failed, setFailed] = useState(earlier?.failed ?? false);
  const [progress, setProgress] = useState(0);
  const [starting, setStarting] = useState(false);
  const [tap] = useState(createTapGuard);
  tap.shown('sensors');

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
    unlockInTap();
    if (phase === 'requesting' || phase === 'measuring') return;
    setFailed(false);
    setPhase('requesting');
    // runSensorCheck measures once the permission is answered: show that as 'measuring'
    void pending.then(() => setPhase((p) => (p === 'requesting' ? 'measuring' : p)));
    const outcome = await beginSensorCheck(pending);
    setPermission(outcome.permission);
    setFailed(outcome.failed);
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

  const onContinue = async () => {
    if (tap.tooSoon() || starting) return;
    unlockInTap(); // inside the tap: the session needs audio and the wake lock
    if (getRepairNext() === 'welcome') {
      navigate('welcome'); // came from "Resume session": its card resumes now that access works
      return;
    }
    setStarting(true);
    try {
      await continueToSession();
    } catch (e) {
      console.error('starting the session failed', e);
      showToast(s.common.errorGeneric, 'error');
      setStarting(false);
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
        <Button testId="btn-continue" size="lg" block disabled={busy || starting || !verdict.canContinue} onClick={() => void onContinue()}>
          {s.common.continue}
        </Button>
      }
    >
      <p class="onb-lead">{t.intro}</p>

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
        variant={phase === 'done' && verdict.canContinue ? 'secondary' : 'primary'}
        disabled={busy}
        onClick={() => void onRequest()}
      >
        {phase === 'done' ? t.measureAgain : t.allow}
      </Button>

      {phase === 'requesting' && <Notice kind="info">{t.requesting}</Notice>}
      {phase === 'measuring' && <ProgressBar value={progress} label={t.measuring} />}

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
