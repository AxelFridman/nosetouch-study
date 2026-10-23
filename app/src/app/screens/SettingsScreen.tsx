/**
 * Settings: language, cue options, simulation mode (testing only, never remembered), the
 * researcher's upload override with a connection test, the active configuration and "Reset this
 * device". Also exports the small upload/confirm helpers shared by the other onboarding/admin screens.
 */
import { useState } from 'preact/hooks';
import { effectiveUpload } from '../../core/config';
import { APP_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION } from '../../core/constants';
import type { Language, UploadConfig, UploadMode } from '../../core/types';
import { LANGUAGES, useStrings, type Strings } from '../../i18n';
import { Button, Card, ChoiceGroup, Field, Notice, Screen, Stat, TextInput, Toggle } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import type { AppState, LocalSettings } from '../state';
import { getState, goBack, navigate, setState, showToast, updateSettings, useAppState } from '../store';

export const UPLOAD_MODES: UploadMode[] = ['apps_script', 'json_post', 'none'];

/** Upload settings in effect (study config + this phone's override). Never throws. */
export function uploadInEffect(st: Pick<AppState, 'config' | 'settings'>): UploadConfig {
  try {
    return effectiveUpload(st.config, st.settings.upload_override);
  } catch (e) {
    // The config module failed: apply the documented overlay rule without its sanitising.
    console.warn('effectiveUpload failed', e);
    return { ...st.config.upload, ...(st.settings.upload_override ?? {}) };
  }
}

export function uploadEnabled(u: UploadConfig): boolean {
  return u.mode !== 'none' && u.endpoint.trim() !== '';
}

/** window.confirm that never throws; false where dialogs are unavailable (some in-app webviews). */
export function confirmDialog(message: string): boolean {
  try {
    return typeof window !== 'undefined' && typeof window.confirm === 'function' ? window.confirm(message) : false;
  } catch {
    return false;
  }
}

/** Same rule as the config sanitizer: an https:// URL, or an absolute path on this site ('/api/ingest'). */
export function isValidEndpoint(endpoint: string): boolean {
  const e = endpoint.trim();
  if (e.startsWith('/')) return !e.startsWith('//'); // '//host' would be protocol-relative, i.e. another site
  try {
    const u = new URL(e);
    return u.protocol === 'https:' && u.hostname !== '';
  } catch {
    return false;
  }
}

export type EndpointProblem = 'required' | 'invalid' | null;

export function endpointProblem(mode: UploadMode, endpoint: string): EndpointProblem {
  const e = endpoint.trim();
  if (e === '') return mode === 'none' ? null : 'required';
  return isValidEndpoint(e) ? null : 'invalid';
}

/** Apps Script web app URLs: /macros/s/<id>/exec, or /a/macros/<domain>/s/<id>/exec for Workspace. */
export function looksLikeAppsScript(endpoint: string): boolean {
  return /^https:\/\/script\.google\.com\/(?:a\/macros\/[^/]+|macros)\/s\/[^/]+\/exec\/?(?:[?#].*)?$/.test(endpoint.trim());
}

/** The override to persist: null when the draft equals the study configuration (nothing to override). */
export function buildOverride(draft: UploadConfig, base: UploadConfig): Partial<UploadConfig> | null {
  const clean: UploadConfig = { mode: draft.mode, endpoint: draft.endpoint.trim(), study_key: draft.study_key.trim() };
  if (clean.mode === base.mode && clean.endpoint === base.endpoint && clean.study_key === base.study_key) return null;
  return clean;
}

export type ProbeResult =
  | { kind: 'ok'; info: string | null }
  | { kind: 'not_ok'; message: string }
  | { kind: 'not_json'; status: number }
  | { kind: 'http'; status: number }
  | { kind: 'network' }
  | { kind: 'timeout' }
  | { kind: 'no_endpoint' };

/**
 * GETs the endpoint. Apps Script's doGet answers {ok: true, ...} through a 302 to
 * script.googleusercontent.com, which is readable cross-origin. Anything that is not ContentService
 * output (dead deployment, re-authorisation needed, uncaught error) is an HTML page without CORS
 * headers, which the browser only reports as a TypeError. No custom headers, so no preflight.
 */
export async function probeEndpoint(
  endpoint: string,
  fetchImpl: typeof fetch = (...args) => fetch(...args),
  timeoutMs = 20000,
): Promise<ProbeResult> {
  const url = endpoint.trim();
  if (url === '') return { kind: 'no_endpoint' };
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl?.abort();
  }, timeoutMs);
  try {
    const res = await fetchImpl(url, {
      method: 'GET',
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'follow',
      signal: ctrl?.signal,
    });
    const text = await res.text();
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      body = undefined;
    }
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      return res.ok ? { kind: 'not_json', status: res.status } : { kind: 'http', status: res.status };
    }
    const b = body as Record<string, unknown>;
    if (b.ok === true) return { kind: 'ok', info: probeInfo(b) };
    if (!res.ok) return { kind: 'http', status: res.status };
    const msg = b.error ?? b.message;
    return { kind: 'not_ok', message: typeof msg === 'string' ? msg : JSON.stringify(b).slice(0, 160) };
  } catch {
    return timedOut ? { kind: 'timeout' } : { kind: 'network' };
  } finally {
    clearTimeout(timer);
  }
}

/** Compact description of what the backend reported besides ok (shown to the researcher). */
function probeInfo(b: Record<string, unknown>): string | null {
  const rest = Object.entries(b).filter(([k, v]) => k !== 'ok' && (typeof v === 'string' || typeof v === 'number'));
  if (rest.length === 0) return null;
  return rest
    .slice(0, 4)
    .map(([k, v]) => `${k}: ${String(v).slice(0, 60)}`)
    .join(', ');
}

export function probeMessage(r: ProbeResult, t: Strings['admin']['settings']['test']): [string, 'ok' | 'warn' | 'error'] {
  switch (r.kind) {
    case 'ok':
      return [r.info ? t.okInfo(r.info) : t.ok, 'ok'];
    case 'not_ok':
      return [t.notOk(r.message), 'error'];
    case 'not_json':
      return [t.notJson(r.status), 'warn'];
    case 'http':
      return [t.http(r.status), 'error'];
    case 'timeout':
      return [t.timeout, 'error'];
    case 'no_endpoint':
      return [t.noEndpoint, 'warn'];
    default:
      return [t.network, 'error'];
  }
}

export function SettingsScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.admin.settings;
  const settings = st.settings;
  const eff = uploadInEffect(st);
  const [draft, setDraft] = useState<UploadConfig>(() => ({ ...eff }));
  const [submitted, setSubmitted] = useState(false);
  const [testing, setTesting] = useState(false);
  const [resetting, setResetting] = useState(false);
  const problem = endpointProblem(draft.mode, draft.endpoint);
  const sessionRunning = st.session?.status === 'in_progress';

  const setCue = (patch: Partial<Pick<LocalSettings, 'audio' | 'speech' | 'vibrate'>>) => {
    updateSettings(patch);
    const cur = getState().settings;
    try {
      getServices().cues.setOptions({ audio: cur.audio, vibrate: cur.vibrate, speech: cur.speech });
    } catch (e) {
      console.warn('cue options not applied', e);
    }
  };

  const onSimulate = (on: boolean) => {
    if (sessionRunning) return;
    const caps = getState().capabilities;
    updateSettings({ simulate: on });
    // A sensor measurement or calibration made with the other source must not carry over into the
    // next session (it would describe a different "device").
    setState({ calibration: null, capabilities: caps ? { ...caps, measured: null } : caps });
  };

  const onSave = () => {
    setSubmitted(true);
    if (problem) {
      showToast(problem === 'required' ? t.errEndpointRequired : t.errEndpoint, 'error');
      return;
    }
    updateSettings({ upload_override: buildOverride(draft, st.config.upload) });
    showToast(t.saved, 'ok');
    try {
      void getServices().uploader.flush();
    } catch (e) {
      console.warn('flush after saving settings failed', e);
    }
  };

  const onDefaults = () => {
    updateSettings({ upload_override: null });
    setDraft({ ...st.config.upload });
    setSubmitted(false);
    showToast(t.defaultsRestored, 'ok');
  };

  const onTest = async () => {
    if (testing) return;
    setTesting(true);
    const result = await probeEndpoint(draft.endpoint.trim() || eff.endpoint);
    setTesting(false);
    const [text, kind] = probeMessage(result, t.test);
    showToast(text, kind, 7000);
  };

  const onReset = async () => {
    if (resetting || !confirmDialog(t.resetConfirm1) || !confirmDialog(t.resetConfirm2)) return;
    setResetting(true);
    try {
      await getServices().store.clearAll();
      updateSettings({ last_participant_id: null });
      setState({
        participant: null,
        session: null,
        calibration: null,
        consentAcceptedAt: null,
        outbox: { pending: 0, sent: 0, failed: 0 },
      });
      showToast(t.resetDone, 'ok');
      navigate('welcome', { returnTo: 'welcome' });
    } catch (e) {
      console.error(e);
      showToast(s.common.errorGeneric, 'error');
      setResetting(false);
    }
  };

  const endpointError =
    submitted && problem ? (problem === 'required' ? t.errEndpointRequired : t.errEndpoint) : null;
  const showAppsScriptWarning =
    draft.mode === 'apps_script' && draft.endpoint.trim() !== '' && !problem && !looksLikeAppsScript(draft.endpoint);

  return (
    <Screen title={t.title} onBack={goBack} testId="screen-settings">
      <Card title={t.language}>
        <ChoiceGroup<Language>
          name="settings-language"
          value={settings.language}
          options={LANGUAGES}
          onChange={(language) => updateSettings({ language })}
          testIdPrefix="opt-lang"
          ariaLabel={t.language}
        />
      </Card>

      <Card title={t.feedbackTitle}>
        <Toggle testId="chk-audio" checked={settings.audio} onChange={(audio) => setCue({ audio })} label={t.audio} hint={t.audioHint} />
        <Toggle testId="chk-speech" checked={settings.speech} onChange={(speech) => setCue({ speech })} label={t.speech} hint={t.speechHint} />
        <Toggle
          testId="chk-vibrate"
          checked={settings.vibrate}
          onChange={(vibrate) => setCue({ vibrate })}
          label={t.vibrate}
          hint={t.vibrateHint}
        />
      </Card>

      <Card title={t.testingTitle} tone={settings.simulate ? 'warn' : undefined}>
        <Toggle
          testId="chk-simulate"
          checked={settings.simulate}
          onChange={onSimulate}
          label={t.simulate}
          hint={sessionRunning ? t.simulateLocked : t.simulateHint}
        />
      </Card>

      <Card title={t.researcherTitle}>
        <div class="stack">
          <p class="small muted">{t.researcherHint}</p>
          <Field label={t.uploadMode}>
            <ChoiceGroup<UploadMode>
              name="settings-upload-mode"
              value={draft.mode}
              columns={1}
              options={UPLOAD_MODES.map((m) => ({ value: m, label: t.uploadModes[m].label, hint: t.uploadModes[m].hint }))}
              onChange={(mode) => setDraft((d) => ({ ...d, mode }))}
              testIdPrefix="opt-uploadmode"
              ariaLabel={t.uploadMode}
            />
          </Field>
          <Field label={t.endpoint} htmlFor="in-endpoint" hint={t.endpointHint} error={endpointError}>
            <div dir="ltr">
              <TextInput
                id="in-endpoint"
                testId="in-endpoint"
                value={draft.endpoint}
                inputMode="url"
                placeholder="https://script.google.com/macros/s/…/exec"
                onInput={(endpoint) => setDraft((d) => ({ ...d, endpoint }))}
              />
            </div>
          </Field>
          {showAppsScriptWarning && <Notice kind="warn">{t.warnAppsScript}</Notice>}
          <Field label={t.studyKey} htmlFor="in-studykey" hint={t.studyKeyHint}>
            <div dir="ltr">
              <TextInput
                id="in-studykey"
                testId="in-studykey"
                value={draft.study_key}
                onInput={(study_key) => setDraft((d) => ({ ...d, study_key }))}
              />
            </div>
          </Field>
          <div class="stack-sm">
            <Button testId="btn-save-settings" block onClick={onSave}>
              {t.save}
            </Button>
            <Button testId="btn-test-upload" variant="secondary" block disabled={testing} onClick={() => void onTest()}>
              {testing ? t.testing : t.testUpload}
            </Button>
            {settings.upload_override && (
              <Button testId="btn-use-defaults" variant="ghost" block onClick={onDefaults}>
                {t.useDefaults}
              </Button>
            )}
          </div>
        </div>
      </Card>

      <Card title={t.effectiveTitle} testId="effective-config">
        <Stat label={t.studyCode} value={<bdi>{st.config.study_code}</bdi>} />
        <Stat label={t.mode} value={t.uploadModes[eff.mode]?.label ?? eff.mode} />
        <Stat
          label={t.endpointShort}
          value={eff.endpoint ? <span class="onb-ltr mono onb-break">{eff.endpoint}</span> : t.notSet}
        />
        <Stat label={t.key} value={eff.study_key ? t.keySet(eff.study_key.length) : t.notSet} />
        <Stat label={t.override} value={settings.upload_override ? s.common.yes : s.common.no} />
        <Stat label={t.consentVersion} value={<bdi>{st.config.consent_version}</bdi>} />
        <Stat label={t.appVersion} value={<bdi>{APP_VERSION}</bdi>} />
        <Stat label={t.schemaVersion} value={<bdi>{SCHEMA_VERSION}</bdi>} />
        <Stat label={t.protocolVersion} value={<bdi>{PROTOCOL_VERSION}</bdi>} />
      </Card>

      <Card title={t.resetTitle} tone="error">
        <div class="stack-sm">
          <p class="small">{t.resetHint}</p>
          <Button testId="btn-reset-device" variant="danger" block disabled={resetting} onClick={() => void onReset()}>
            {t.reset}
          </Button>
        </div>
      </Card>
    </Screen>
  );
}
