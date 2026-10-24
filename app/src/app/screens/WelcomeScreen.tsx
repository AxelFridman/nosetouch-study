/**
 * Welcome + consent (spec v2 §1, screen 1): language switch, a small illustration of the gesture, the
 * four key facts of the participant information, the full information in a <details>, environment
 * warnings (insecure context, no motion sensors, in-app browser, simulation, researcher mode, pending
 * uploads), "Continue as {name}" + "New person" when a profile exists on this phone, resuming an
 * unfinished session, and the menu. The consent checkbox and Start sit in the sticky footer.
 *
 * The Start tap (startFlow.startFromWelcome, order pinned by tests/startFlow.test.ts): requests the
 * motion permission as its FIRST synchronous statement (WebKit only shows the prompt during the
 * gesture), unlocks audio and takes the wake lock (iOS needs a gesture for both), records the consent
 * time, starts the background sensor measurement (sensorCheck.ts) and opens the profile.
 * NOTE for the PI: the wording is consent_version in public/config.json (2026-10-v2).
 */
import type { ComponentChildren } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { localizedText } from '../../core/config';
import { APP_VERSION } from '../../core/constants';
import type { Language, SessionRecord } from '../../core/types';
import { LANGUAGES, useStrings } from '../../i18n';
import { Button, Card, ChoiceGroup, Notice, Screen, Toggle } from '../../ui/components';
import { GestureAnimation } from '../../ui/GestureAnimation';
import '../../ui/onboarding.css';
import { patchCapabilities, safeRequestPermission, setRepairNext } from '../sensorCheck';
import { getServices } from '../services';
import { resumeSession } from '../sessionFlow';
import { forgetParticipant, startFromWelcome, unlockInTap } from '../startFlow';
import type { ScreenId } from '../state';
import { getState, navigate, setState, showToast, updateSettings, useAppState } from '../store';
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

/** mailto: only for something that looks like a single e-mail address (config is editable text). */
export function mailHref(email: string): string | null {
  return /^[^\s@<>"']+@[^\s@<>"']+\.[^\s@<>"']+$/.test(email) ? `mailto:${email}` : null;
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
  const c = s.onboarding.consent;
  const cfg = st.config;
  const participant = st.participant;
  const pid = participant?.participant_id ?? null;
  const [resumable, setResumable] = useState<SessionRecord | null>(null);
  const [resuming, setResuming] = useState(false);
  // consent is per app run: coming back from the profile keeps the box ticked
  const [agreed, setAgreed] = useState(() => getState().consentAcceptedAt !== null);

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

  const onAgree = (v: boolean) => {
    setAgreed(v);
    if (!v) setState({ consentAcceptedAt: null }); // un-ticked: no consent until ticked again
  };

  // Synchronous: its FIRST statement is the motion-permission request (WebKit shows the prompt only
  // during the gesture), then audio + wake lock, consent time, background sensor check, profile.
  // Never starts without consent (the button is disabled then anyway).
  const onStart = () => void startFromWelcome(agreed);

  const onNewPerson = () => {
    forgetParticipant();
    setAgreed(false);
  };

  const onResume = async () => {
    // FIRST statement, before any await: WebKit only shows the motion prompt during the gesture, and
    // the grant must be requested again after every page load. Audio and wake lock need a tap too.
    const permission = safeRequestPermission();
    unlockInTap();
    if (!resumable || resuming) return;
    setResuming(true);
    try {
      let perm = await permission;
      if (getState().settings.simulate) perm = 'granted';
      patchCapabilities({ permission_state: perm });
      if (perm === 'denied') {
        setRepairNext('welcome'); // the repair screen's Continue comes back to this resume card
        navigate('sensors'); // first: navigate() clears the toast
        showToast(t.resumeNeedsSensors, 'warn', 6000);
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
  const uploads = uploadEnabled(upload);
  const pending = st.outbox.pending + st.outbox.failed;
  const inApp = inAppName(st.device?.in_app_browser);
  const locale = s.admin.locale;
  const institution = localizedText(cfg.institution, st.settings.language);
  const region = localizedText(cfg.region, st.settings.language);
  const contactName = cfg.contact.name.trim();
  const contactEmail = cfg.contact.email.trim();
  const href = mailHref(contactEmail);
  const askPermission = !!caps?.permission_api && caps.permission_state !== 'granted' && !st.settings.simulate;

  const section = (title: string, body: ComponentChildren) => (
    <section class="onb-section">
      <h3>{title}</h3>
      {body}
    </section>
  );

  return (
    <Screen
      title={t.title}
      testId="screen-welcome"
      footer={
        <>
          <div class="onb-agree">
            <Toggle testId="chk-consent" checked={agreed} onChange={onAgree} label={t.agree(cfg.min_age, region)} />
          </div>
          {askPermission && agreed && <p class="small muted onb-footer-hint">{t.permissionNote}</p>}
          <Button testId="btn-start" size="lg" block disabled={!agreed} onClick={onStart}>
            {t.start}
          </Button>
        </>
      }
    >
      <div class="onb-lang">
        <ChoiceGroup<Language>
          name="welcome-language"
          value={st.settings.language}
          options={LANGUAGES}
          onChange={(language) => updateSettings({ language })}
          testIdPrefix="opt-lang"
          ariaLabel={t.language}
        />
      </div>

      <div class="onb-hero">
        <div class="onb-hero-gesture" data-testid="welcome-gesture">
          <GestureAnimation
            hand={null}
            orientation="upright_toward"
            speed="natural"
            contact="any"
            view="side"
            compact
            label={t.gestureLabel}
          />
        </div>
        <p class="onb-tagline">{t.lead}</p>
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
      {st.setupRequested && (
        <Notice kind="info" testId="warn-setup">
          {t.warnSetup}
        </Notice>
      )}
      {pending > 0 && uploads && (
        <Notice kind="info" testId="pending-uploads">
          {t.pendingUploads(pending)}
        </Notice>
      )}

      {participant && (
        <Card tone="info" testId="returning-card">
          <div class="stack-sm">
            <p class="onb-returning-name">
              {t.continueAs(participant.nickname?.trim() || participant.participant_id)}
            </p>
            <p class="small muted">
              <bdi class="mono">{participant.participant_id}</bdi> · {t.continueHint}
            </p>
            <Button testId="btn-new-participant" variant="secondary" block onClick={onNewPerson}>
              {t.newPerson}
            </Button>
          </div>
        </Card>
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

      <Card testId="consent-summary">
        <ul class="onb-summary">
          <li>{t.summaryDuration}</li>
          <li>{t.summarySensors}</li>
          <li>{t.summaryStored(uploads, cfg.upload_nickname)}</li>
          <li>{t.summaryDelete}</li>
        </ul>
        {contactName || contactEmail ? (
          <p class="small onb-contact" data-testid="consent-contact">
            {c.contactIntro} <bdi>{contactName}</bdi>
            {contactName && contactEmail ? ' · ' : ''}
            {contactEmail &&
              (href ? (
                <a href={href} class="onb-ltr">
                  {contactEmail}
                </a>
              ) : (
                <bdi>{contactEmail}</bdi>
              ))}
          </p>
        ) : (
          <p class="small onb-contact" data-testid="consent-contact">
            {c.contactNone}
          </p>
        )}
        <details class="onb-details" data-testid="consent-details">
          <summary>{t.fullInfo}</summary>
          <div class="onb-consent">
            {section(c.purposeTitle, <p>{c.purpose(institution)}</p>)}
            {section(c.procedureTitle, <p>{c.procedure}</p>)}
            {section(c.durationTitle, <p>{c.duration}</p>)}
            {section(
              c.risksTitle,
              <ul>
                {c.risks.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>,
            )}
            {section(c.voluntaryTitle, <p>{c.voluntary}</p>)}
            {section(
              c.collectedTitle,
              <ul>
                {c.collected.map((x) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>,
            )}
            {section(
              c.notCollectedTitle,
              <>
                <p>{c.notCollected}</p>
                <p>{cfg.upload_nickname && uploads ? c.nameUploaded : c.nameLocal}</p>
              </>,
            )}
            {section(c.storageTitle, <p>{uploads ? c.storageUpload : c.storageLocal}</p>)}
            {section(c.retentionTitle, <p>{c.retention(institution)}</p>)}
            <p class="small muted">{c.version(cfg.consent_version)}</p>
          </div>
        </details>
      </Card>

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
