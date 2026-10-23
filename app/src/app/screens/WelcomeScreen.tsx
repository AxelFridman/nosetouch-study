/**
 * Welcome: what the study is, language switch, environment warnings (insecure context, no motion
 * sensors, in-app browser, simulation, pending uploads), start / continue as the known participant /
 * new participant, resume an unfinished session, and the menu.
 */
import { useEffect, useState } from 'preact/hooks';
import { APP_VERSION } from '../../core/constants';
import type { Language, SessionRecord } from '../../core/types';
import { LANGUAGES, useStrings } from '../../i18n';
import { Button, Card, ChoiceGroup, Field, Notice, Screen } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { resumeSession } from '../sessionFlow';
import type { ScreenId } from '../state';
import { getState, navigate, setState, showToast, updateSettings, useAppState } from '../store';
import { patchCapabilities, safeRequestPermission } from './SensorCheckScreen';
import { uploadEnabled, uploadInEffect } from './SettingsScreen';

/** in_app_browser ids reported by core/sensors → display names ('' = use the generic wording). */
const IN_APP_NAMES: Record<string, string> = {
  instagram: 'Instagram',
  facebook: 'Facebook',
  tiktok: 'TikTok',
  snapchat: 'Snapchat',
  line: 'LINE',
  google_app: 'Google',
  linkedin: 'LinkedIn',
  wechat: 'WeChat',
  pinterest: 'Pinterest',
  android_webview: '',
  ios_webview: '',
};

/** Display name of a detected in-app browser; null when none, '' for an unnamed webview. */
export function inAppName(id: string | null | undefined): string | null {
  if (!id) return null;
  if (Object.prototype.hasOwnProperty.call(IN_APP_NAMES, id)) return IN_APP_NAMES[id];
  return id.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatWhen(iso: string, locale: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  try {
    return d.toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return d.toISOString().slice(0, 16).replace('T', ' ');
  }
}

const MENU: { screen: ScreenId; testId: string; key: 'menuData' | 'menuSettings' | 'menuSandbox' | 'menuAbout' }[] = [
  { screen: 'data', testId: 'btn-menu-data', key: 'menuData' },
  { screen: 'settings', testId: 'btn-menu-settings', key: 'menuSettings' },
  { screen: 'sandbox', testId: 'btn-menu-sandbox', key: 'menuSandbox' },
  { screen: 'about', testId: 'btn-menu-about', key: 'menuAbout' },
];

export function WelcomeScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.welcome;
  const participant = st.participant;
  const pid = participant?.participant_id ?? null;
  const [resumable, setResumable] = useState<SessionRecord | null>(null);
  const [resuming, setResuming] = useState(false);

  useEffect(() => {
    setResumable(null);
    if (!pid) return;
    let alive = true;
    void (async () => {
      try {
        const found = await getServices().store.findResumableSession(pid);
        if (alive) setResumable(found ?? null);
      } catch (e) {
        console.warn('looking up a resumable session failed', e);
      }
    })();
    return () => {
      alive = false;
    };
  }, [pid]);

  const onStart = () => {
    // Consent is per app run: a known participant who already agreed in this run goes on to check
    // their profile; everyone else reads the consent first.
    navigate(getState().consentAcceptedAt && getState().participant ? 'profile' : 'consent');
  };

  const onNewParticipant = () => {
    // Someone else: they must consent themselves, and nothing of the previous person may carry over.
    setState({ participant: null, calibration: null, consentAcceptedAt: null, session: null });
    navigate('consent');
  };

  const onResume = async () => {
    // FIRST statement, before any await: WebKit only shows the motion prompt during the gesture, and
    // the grant must be requested again after every page load. Audio and wake lock need a tap too.
    const permission = safeRequestPermission();
    try {
      const services = getServices();
      void services.cues.unlock();
      void services.wakeLock.acquire();
    } catch (e) {
      console.warn('audio / wake lock unavailable', e);
    }
    if (!resumable || resuming) return;
    setResuming(true);
    try {
      let perm = await permission;
      if (getState().settings.simulate) perm = 'granted';
      patchCapabilities({ permission_state: perm });
      if (perm === 'denied') {
        showToast(t.resumeNeedsSensors, 'warn', 6000);
        navigate('sensors');
        return;
      }
      const session = await resumeSession(resumable.session_id);
      if (session) {
        navigate('session');
        return;
      }
      setResumable(null);
      showToast(t.resumeFailed, 'warn');
    } catch (e) {
      console.error('resuming the session failed', e);
      showToast(s.common.errorGeneric, 'error');
    }
    setResuming(false);
  };

  const caps = st.capabilities;
  const upload = uploadInEffect(st);
  const pending = st.outbox.pending + st.outbox.failed;
  const inApp = inAppName(st.device?.in_app_browser);
  const locale = s.admin.locale;

  return (
    <Screen
      title={s.common.appName}
      testId="screen-welcome"
      footer={
        <>
          <Button testId="btn-start" size="lg" block onClick={onStart}>
            {participant ? (
              <span>
                {t.continueAs} <bdi class="onb-nowrap">{participant.participant_id}</bdi>
                {participant.nickname ? (
                  <>
                    {' '}
                    (<bdi>{participant.nickname}</bdi>)
                  </>
                ) : null}
              </span>
            ) : (
              t.start
            )}
          </Button>
          {participant && (
            <Button testId="btn-new-participant" variant="ghost" block onClick={onNewParticipant}>
              {t.newParticipant}
            </Button>
          )}
        </>
      }
    >
      <div class="onb-hero">
        <img class="onb-hero-icon" src="./icons/icon.svg" alt="" width={72} height={72} />
        <p class="onb-tagline">{t.tagline}</p>
      </div>

      {caps && !caps.secure_context && (
        <Notice kind="error" testId="warn-insecure">
          {t.warnInsecure}
        </Notice>
      )}
      {caps && caps.secure_context && !caps.device_motion && !st.settings.simulate && (
        <Notice kind="error" testId="warn-no-motion">
          {t.warnNoMotion}
        </Notice>
      )}
      {inApp !== null && (
        <Notice kind="warn" testId="warn-in-app">
          {inApp ? t.warnInApp(inApp) : t.warnInAppGeneric}
        </Notice>
      )}
      {st.settings.simulate && (
        <Notice kind="warn" testId="warn-sim">
          {t.warnSim}
        </Notice>
      )}
      {pending > 0 && uploadEnabled(upload) && (
        <Notice kind="info" testId="pending-uploads">
          {t.pendingUploads(pending)}
        </Notice>
      )}

      {resumable && (
        <Card title={t.resumeTitle} tone="warn" testId="resume-card">
          <div class="stack-sm">
            <p>
              {t.resumeBody(resumable.progress.n_accepted, resumable.protocol.n_planned_trials)}{' '}
              {t.resumeStarted(formatWhen(resumable.started_at, locale))}
            </p>
            <Button testId="btn-resume-session" block disabled={resuming} onClick={() => void onResume()}>
              {resuming ? t.resuming : t.resume}
            </Button>
          </div>
        </Card>
      )}

      <Card title={t.aboutTitle}>
        <p>{t.aboutBody}</p>
        <h3 class="onb-subhead">{t.needsTitle}</h3>
        <ul class="onb-checklist">
          {t.needs.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
        <p class="small muted onb-privacy">{t.privacy}</p>
      </Card>

      <Field label={t.language}>
        <ChoiceGroup<Language>
          name="welcome-language"
          value={st.settings.language}
          options={LANGUAGES}
          onChange={(language) => updateSettings({ language })}
          testIdPrefix="opt-lang"
          ariaLabel={t.language}
        />
      </Field>

      <nav class="onb-menu" aria-label={t.menuTitle}>
        {MENU.map((m) => (
          <Button key={m.screen} testId={m.testId} variant="secondary" onClick={() => navigate(m.screen, { returnTo: 'welcome' })}>
            {t[m.key]}
          </Button>
        ))}
      </nav>

      <p class="small muted center">{t.version(APP_VERSION)}</p>
    </Screen>
  );
}
