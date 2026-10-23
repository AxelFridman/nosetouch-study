/**
 * Participant information + consent. The texts depend on the study configuration: minimum age,
 * contact, whether a nickname is uploaded and whether data is uploaded at all. Accepting stores the
 * time in state.consentAcceptedAt (copied into the session record with config.consent_version).
 * NOTE for the PI: the wording must match the version approved by the ethics committee; bump
 * consent_version in public/config.json whenever it changes.
 */
import type { ComponentChildren } from 'preact';
import { useState } from 'preact/hooks';
import { localizedText } from '../../core/config';
import { isoTime } from '../../core/util';
import { useStrings } from '../../i18n';
import { Button, Card, Screen, Toggle } from '../../ui/components';
import '../../ui/onboarding.css';
import { navigate, setState, useAppState } from '../store';
import { uploadEnabled, uploadInEffect } from './SettingsScreen';

/** mailto: only for something that looks like a single e-mail address (config is editable text). */
function mailHref(email: string): string | null {
  return /^[^\s@<>"']+@[^\s@<>"']+\.[^\s@<>"']+$/.test(email) ? `mailto:${email}` : null;
}

export function ConsentScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.consent;
  const cfg = st.config;
  const institution = localizedText(cfg.institution, st.settings.language);
  const region = localizedText(cfg.region, st.settings.language);
  const [agreed, setAgreed] = useState(false);
  const uploads = uploadEnabled(uploadInEffect(st));
  const name = cfg.contact.name.trim();
  const email = cfg.contact.email.trim();
  const href = mailHref(email);

  const accept = () => {
    if (!agreed) return;
    setState({ consentAcceptedAt: isoTime() });
    navigate('profile');
  };

  const section = (title: string, body: ComponentChildren) => (
    <section class="onb-section">
      <h2>{title}</h2>
      {body}
    </section>
  );

  return (
    <Screen title={t.title} onBack={() => navigate('welcome')} testId="screen-consent">
      <p class="onb-lead">{t.intro}</p>
      <div class="onb-consent">
        {section(t.purposeTitle, <p>{t.purpose(institution)}</p>)}
        {section(t.procedureTitle, <p>{t.procedure}</p>)}
        {section(t.durationTitle, <p>{t.duration}</p>)}
        {section(
          t.risksTitle,
          <ul>
            {t.risks.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>,
        )}
        {section(t.voluntaryTitle, <p>{t.voluntary}</p>)}
        {section(
          t.collectedTitle,
          <ul>
            {t.collected.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>,
        )}
        {section(
          t.notCollectedTitle,
          <>
            <p>{t.notCollected}</p>
            <p>{cfg.upload_nickname ? t.nicknameUploaded : t.nicknameLocal}</p>
          </>,
        )}
        {section(t.storageTitle, <p>{uploads ? t.storageUpload : t.storageLocal}</p>)}
        {section(t.retentionTitle, <p>{t.retention(institution)}</p>)}
        {section(
          t.contactTitle,
          name || email ? (
            <p data-testid="consent-contact">
              {t.contactIntro} <bdi>{name}</bdi>
              {name && email ? ' · ' : ''}
              {email &&
                (href ? (
                  <a href={href} class="onb-ltr">
                    {email}
                  </a>
                ) : (
                  <bdi>{email}</bdi>
                ))}
            </p>
          ) : (
            <p>{t.contactNone}</p>
          ),
        )}
      </div>

      <Card tone="info">
        <Toggle testId="chk-consent" checked={agreed} onChange={setAgreed} label={t.agree(cfg.min_age, region)} />
      </Card>
      <div class="stack-sm">
        <Button testId="btn-consent-accept" size="lg" block disabled={!agreed} onClick={accept}>
          {t.accept}
        </Button>
        <Button testId="btn-consent-decline" variant="ghost" block onClick={() => navigate('welcome')}>
          {t.decline}
        </Button>
      </div>
      <p class="small muted">{t.version(cfg.consent_version)}</p>
    </Screen>
  );
}
