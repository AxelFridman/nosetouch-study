/**
 * About: what the study is, how the data is handled, contact, how to install the app, versions and
 * a link to the sensor playground (sandbox).
 */
import { localizedText } from '../../core/config';
import { APP_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION } from '../../core/constants';
import { useStrings } from '../../i18n';
import { Button, Card, Screen, Stat } from '../../ui/components';
import '../../ui/onboarding.css';
import { getState, goBack, navigate, useAppState } from '../store';

export function AboutScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.admin.about;
  const name = st.config.contact.name.trim();
  const email = st.config.contact.email.trim();
  const mailOk = /^[^\s@<>"']+@[^\s@<>"']+\.[^\s@<>"']+$/.test(email);

  // Keep the original "back" target: the sandbox's back button then returns to where About came from.
  const openSandbox = () => navigate('sandbox', { returnTo: getState().returnTo === 'about' ? 'welcome' : getState().returnTo });

  return (
    <Screen title={t.title} onBack={goBack} testId="screen-about">
      <div class="onb-hero">
        <img class="onb-hero-icon" src="./icons/icon.svg" alt="" width={64} height={64} />
        <p class="onb-tagline">{s.common.appName}</p>
      </div>

      <Card title={t.whatTitle}>
        <p>{t.what(localizedText(st.config.institution, st.settings.language))}</p>
        <h3 class="onb-subhead">{t.stagesTitle}</h3>
        <ol class="onb-steps">
          {t.stages.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ol>
      </Card>

      <Card title={t.dataTitle}>
        <ul class="onb-checklist">
          {t.data.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Card>

      <Card title={t.contactTitle} testId="about-contact">
        {name || email ? (
          <p>
            {name && <bdi>{name}</bdi>}
            {name && email ? ' · ' : ''}
            {email &&
              (mailOk ? (
                <a href={`mailto:${email}`} class="onb-ltr">
                  {email}
                </a>
              ) : (
                <bdi>{email}</bdi>
              ))}
          </p>
        ) : (
          <p>{t.contactNone}</p>
        )}
      </Card>

      <Card title={t.installTitle}>
        <ul class="onb-checklist">
          <li>{t.installIos}</li>
          <li>{t.installAndroid}</li>
        </ul>
        <p class="small muted">{t.installWhy}</p>
      </Card>

      <Card title={t.sandboxTitle}>
        <div class="stack-sm">
          <p>{t.sandbox}</p>
          <Button testId="btn-open-sandbox" variant="secondary" block onClick={openSandbox}>
            {t.openSandbox}
          </Button>
        </div>
      </Card>

      <Card title={t.versionsTitle} testId="about-versions">
        <Stat label={s.admin.settings.appVersion} value={<bdi>{APP_VERSION}</bdi>} />
        <Stat label={s.admin.settings.schemaVersion} value={<bdi>{SCHEMA_VERSION}</bdi>} />
        <Stat label={s.admin.settings.protocolVersion} value={<bdi>{PROTOCOL_VERSION}</bdi>} />
        <Stat label={s.admin.settings.consentVersion} value={<bdi>{st.config.consent_version}</bdi>} />
        <Stat label={t.studyCode} value={<bdi>{st.config.study_code}</bdi>} />
      </Card>
    </Screen>
  );
}
