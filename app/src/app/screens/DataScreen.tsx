/**
 * My data: the sessions and participants stored on this phone, per-session upload state, export
 * (one session or everything), "upload now", storage use, and deletion (the right to withdraw).
 * Uploaded copies can only be deleted by the research team; the screen says so.
 * Items the server refused (OutboxItem.permanent, e.g. a study key that was changed since this page
 * loaded config.json) are never retried automatically: a warning asks to reload, then "Upload now".
 */
import { useEffect, useState } from 'preact/hooks';
import { redactProfile } from '../../core/config';
import { deliverExport } from '../../core/exporter';
import type { OutboxItem, ParticipantProfile, SessionRecord } from '../../core/types';
import { useStrings, type Strings } from '../../i18n';
import { Button, Card, Notice, Screen, Spinner } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { getState, goBack, setState, showToast, updateSettings, useAppState } from '../store';
import { formatWhen } from './WelcomeScreen';
import { confirmDialog, uploadEnabled, uploadInEffect } from './SettingsScreen';

export interface UploadSummary {
  total: number;
  sent: number;
  pending: number;
  /** failed for now; retried automatically */
  failed: number;
  /** refused by the server; sent again only by "Upload now" */
  refused: number;
}

/** An outbox item the server refused: the automatic retries skip it (core/uploader.ts). */
export function isRefused(item: OutboxItem): boolean {
  return item.status === 'failed' && item.permanent === true;
}

/** Outbox state of one session's records (the session record itself and its trials). */
export function summarizeOutbox(items: OutboxItem[], sessionId: string): UploadSummary {
  const out: UploadSummary = { total: 0, sent: 0, pending: 0, failed: 0, refused: 0 };
  for (const it of items) {
    if (it.session_id !== sessionId) continue;
    out.total += 1;
    if (isRefused(it)) out.refused += 1;
    else out[it.status] += 1;
  }
  return out;
}

export function uploadLabel(u: UploadSummary, enabled: boolean, t: Strings['admin']['data']['uploadState']): string {
  if (u.total === 0) return enabled ? t.none : t.off;
  if (u.sent === u.total) return t.done;
  if (!enabled) return t.off;
  if (u.refused > 0) return t.refused(u.refused);
  if (u.failed > 0) return t.failed(u.failed + u.pending);
  return t.partial(u.sent, u.total);
}

/** "1.2 MB"-style sizes (decimal units, like the OS storage screens). */
export function formatBytes(n: number, locale = 'en'): string {
  const units = ['B', 'kB', 'MB', 'GB', 'TB'];
  let v = Math.max(0, n);
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i += 1;
  }
  const digits = i === 0 || v >= 100 ? 0 : 1;
  let num: string;
  try {
    num = v.toLocaleString(locale, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
  } catch {
    num = v.toFixed(digits);
  }
  return `${num} ${units[i]}`;
}

interface Loaded {
  sessions: SessionRecord[];
  outbox: OutboxItem[];
  participants: ParticipantProfile[];
  estimate: { usage: number; quota: number } | null;
}

export function DataScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.admin.data;
  const locale = s.admin.locale;
  const [data, setData] = useState<Loaded | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const upload = uploadInEffect(st);
  const enabled = uploadEnabled(upload);

  const reload = async () => {
    try {
      const store = getServices().store;
      const [sessions, outbox, participants, estimate] = await Promise.all([
        store.listSessions(),
        store.listOutbox(),
        store.listParticipants(),
        store.estimate().catch(() => null),
      ]);
      setData({ sessions, outbox, participants, estimate });
      setLoadFailed(false);
    } catch (e) {
      console.error('loading the local data failed', e);
      setLoadFailed(true);
    }
  };

  // Reload when the upload queue changes (the uploader updates state.outbox while it sends).
  useEffect(() => {
    void reload();
  }, [st.outbox.pending, st.outbox.sent, st.outbox.failed]);

  const refreshCounts = async () => {
    try {
      setState({ outbox: await getServices().store.outboxCounts() });
    } catch {
      // counts are informative only
    }
  };

  const redact = (p: ParticipantProfile) => {
    try {
      return redactProfile(p, getState().config);
    } catch {
      // never export a nickname the configuration did not allow to leave the phone
      return getState().config.upload_nickname ? p : { ...p, nickname: null };
    }
  };

  const onExport = async (sessionIds?: string[]) => {
    if (busy) return;
    setBusy(sessionIds ? `export:${sessionIds[0]}` : 'export');
    try {
      const bundle = await getServices().store.exportBundle(sessionIds ? { sessionIds, redact } : { redact });
      const res = await deliverExport(bundle);
      if (res === 'shared') showToast(t.exportShared, 'ok');
      else if (res === 'downloaded') showToast(t.exportDownloaded, 'ok');
      else showToast(t.exportCancelled, 'info');
    } catch (e) {
      console.error('export failed', e);
      showToast(t.exportFailed, 'error');
    }
    setBusy(null);
  };

  const onUploadNow = async () => {
    if (busy) return;
    if (!enabled) {
      showToast(t.uploadOff, 'warn', 6000);
      return;
    }
    setBusy('upload');
    try {
      const r = await getServices().uploader.flush({ force: true });
      setState({ outbox: r.counts });
      if (r.attempted === 0) showToast(t.uploadNothing, 'info');
      else if (r.failed === 0) showToast(t.uploadDone(r.sent), 'ok');
      else if ((r.refused ?? 0) > 0) showToast(t.uploadRefused(r.refused ?? 0), 'warn', 8000);
      else if (r.sent > 0) showToast(t.uploadSomeFailed(r.sent, r.failed), 'warn', 6000);
      else showToast(t.uploadFailed, 'error', 6000);
    } catch (e) {
      console.error('upload failed', e);
      showToast(t.uploadFailed, 'error', 6000);
    }
    setBusy(null);
    void reload();
  };

  const onDeleteSession = async (session: SessionRecord) => {
    if (busy || !confirmDialog(t.confirmDeleteSession(formatWhen(session.started_at, locale)))) return;
    setBusy(`delete:${session.session_id}`);
    try {
      await getServices().store.deleteSession(session.session_id);
      if (getState().session?.session_id === session.session_id) setState({ session: null });
      showToast(t.deletedSession, 'ok');
    } catch (e) {
      console.error('deleting the session failed', e);
      showToast(s.common.errorGeneric, 'error');
    }
    setBusy(null);
    await refreshCounts();
    await reload();
  };

  const onDeleteParticipant = async (p: ParticipantProfile, nSessions: number) => {
    if (busy || !confirmDialog(t.confirmDeleteParticipant(p.participant_id, nSessions))) return;
    setBusy(`participant:${p.participant_id}`);
    try {
      await getServices().store.deleteParticipant(p.participant_id);
      const now = getState();
      if (now.participant?.participant_id === p.participant_id) {
        setState({ participant: null, calibration: null, consentAcceptedAt: null });
      }
      if (now.session?.participant_id === p.participant_id) setState({ session: null });
      if (now.settings.last_participant_id === p.participant_id) updateSettings({ last_participant_id: null });
      showToast(t.deletedParticipant, 'ok');
    } catch (e) {
      console.error('deleting the participant failed', e);
      showToast(s.common.errorGeneric, 'error');
    }
    setBusy(null);
    await refreshCounts();
    await reload();
  };

  const sessions = data?.sessions ?? [];
  const counts = st.outbox;
  const refused = (data?.outbox ?? []).filter(isRefused).length;
  const contact = [st.config.contact.name.trim(), st.config.contact.email.trim()].filter(Boolean).join(' · ');

  return (
    <Screen title={t.title} onBack={goBack} testId="screen-data">
      <p class="onb-lead">{t.intro}</p>
      {loadFailed && <Notice kind="error">{s.common.errorGeneric}</Notice>}
      {!enabled && <Notice kind="info">{t.uploadOff}</Notice>}
      {enabled && refused > 0 && (
        <Notice kind="warn" testId="notice-refused">
          {t.refused(refused)}
        </Notice>
      )}

      <div class="onb-actions">
        <Button
          testId="btn-export-all"
          block
          disabled={busy !== null || sessions.length === 0}
          onClick={() => void onExport()}
        >
          {busy === 'export' ? t.busy : t.exportAll}
        </Button>
        <Button testId="btn-upload-now" variant="secondary" block disabled={busy !== null} onClick={() => void onUploadNow()}>
          {busy === 'upload' ? t.busy : t.uploadNow}
        </Button>
      </div>

      <section class="stack-sm" aria-labelledby="data-sessions">
        <h2 id="data-sessions">{t.sessionsTitle}</h2>
        {!data && !loadFailed && <Spinner label={s.common.loading} />}
        {data && sessions.length === 0 && <p class="muted">{t.noSessions}</p>}
        <ul class="clean stack-sm">
          {sessions.map((sess) => {
            const u = summarizeOutbox(data?.outbox ?? [], sess.session_id);
            const deleting = busy === `delete:${sess.session_id}`;
            const exporting = busy === `export:${sess.session_id}`;
            return (
              <li key={sess.session_id} class="card onb-row" data-testid="session-row">
                <div class="onb-row-head">
                  <strong>{formatWhen(sess.started_at, locale)}</strong>
                  <span class={`badge onb-status-${sess.status}`}>{t.status[sess.status]}</span>
                  {sess.simulated && <span class="badge badge-sim">{t.simulated}</span>}
                </div>
                <div class="small">
                  <bdi class="mono">{sess.participant_id}</bdi> · {s.onboarding.setup.presets[sess.protocol.preset]?.label ?? sess.protocol.preset}{' '}
                  · {t.movements(sess.progress.n_accepted, sess.protocol.n_planned_trials)}
                </div>
                <div class="small muted" data-testid="session-upload-state">
                  {uploadLabel(u, enabled, t.uploadState)}
                </div>
                <div class="onb-row-actions">
                  <Button
                    testId="btn-export-session"
                    variant="secondary"
                    disabled={busy !== null}
                    onClick={() => void onExport([sess.session_id])}
                  >
                    {exporting ? t.busy : t.exportSession}
                  </Button>
                  <Button testId="btn-delete-session" variant="ghost" disabled={busy !== null} onClick={() => void onDeleteSession(sess)}>
                    {deleting ? t.busy : t.deleteSession}
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      {enabled && (
        <Card title={t.queueTitle}>
          <p class="small">{t.queue(counts.pending, Math.max(0, counts.failed - refused), counts.sent, refused)}</p>
        </Card>
      )}

      <Card title={t.participantsTitle} testId="participants">
        {data && data.participants.length === 0 && <p class="muted">{t.noParticipants}</p>}
        <ul class="clean stack-sm">
          {(data?.participants ?? []).map((p) => {
            const n = sessions.filter((x) => x.participant_id === p.participant_id).length;
            return (
              <li key={p.participant_id} class="onb-participant" data-testid="participant-row">
                <div>
                  <bdi class="mono">{p.participant_id}</bdi>
                  {p.nickname ? (
                    <>
                      {' '}
                      (<bdi>{p.nickname}</bdi>)
                    </>
                  ) : null}
                  <div class="small muted">
                    {t.since(formatWhen(p.created_at, locale))} · {t.sessionsCount(n)}
                  </div>
                </div>
                <Button
                  testId="btn-delete-participant"
                  variant="danger"
                  block
                  disabled={busy !== null}
                  onClick={() => void onDeleteParticipant(p, n)}
                >
                  {busy === `participant:${p.participant_id}` ? t.busy : t.deleteParticipant}
                </Button>
              </li>
            );
          })}
        </ul>
        <p class="small muted onb-note">
          {t.uploadedNote}
          {contact && (
            <>
              {' '}
              {t.contact} <bdi>{contact}</bdi>
            </>
          )}
        </p>
      </Card>

      <Card title={t.storageTitle} testId="storage">
        <div class="stack-sm">
          {data?.estimate && (
            <p data-testid="storage-estimate">
              {t.storageUsed(formatBytes(data.estimate.usage, locale), formatBytes(data.estimate.quota, locale))}
            </p>
          )}
          <p class="small muted">{t.storageTip}</p>
        </div>
      </Card>
    </Screen>
  );
}
