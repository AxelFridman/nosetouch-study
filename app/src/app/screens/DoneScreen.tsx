/**
 * Done (spec v2 §1, screen 4): "Thank you, {first name}!", how many movements were recorded, the live
 * upload state of the outbox (state.outbox, updated by the upload queue: all sent / sending, keep the
 * page open / offline), the participant code (needed to have the data deleted later), "Next person"
 * (the phone goes to someone else: nothing of this participant stays on the welcome screen) and Home.
 * For researchers: attempts and time, "Upload now" and the "Save / share my data" export.
 *
 * The export bundle is prepared as soon as the screen opens, so that the tap calls deliverExport
 * synchronously: iOS only opens the share sheet inside a user gesture, which an IndexedDB read
 * before navigator.share() would outlive. If the bundle is not ready yet, it is built on the tap.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type { ExportBundle, SessionRecord, UploadConfig } from '../../core/types';
import { effectiveUpload, redactProfile } from '../../core/config';
import { deliverExport } from '../../core/exporter';
import { Button, Card, Notice, Screen, Stat } from '../../ui/components';
import { useStrings, type Strings } from '../../i18n';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { sessionMinutes } from '../sessionFlow';
import { firstName, forgetParticipant } from '../startFlow';
import { getState, navigate, setState, showToast, useAppState } from '../store';

function buildBundle(sessionId: string): Promise<ExportBundle> {
  return getServices().store.exportBundle({
    sessionIds: [sessionId],
    redact: (p) => redactProfile(p, getState().config),
  });
}

export function uploadConfigured(cfg: UploadConfig | null): boolean {
  return !!cfg && cfg.mode !== 'none' && cfg.endpoint.trim() !== '';
}

/** What the upload card says. `left` = items not sent yet (pending + failed, refused included). */
export type DoneUpload =
  | { kind: 'off' }
  | { kind: 'all_sent' }
  | { kind: 'sending'; left: number; failed: number }
  | { kind: 'offline'; left: number };

/** The upload card's state from the outbox counters (pure; the screen re-renders on every change). */
export function doneUploadState(input: { configured: boolean; pending: number; failed: number; online: boolean }): DoneUpload {
  if (!input.configured) return { kind: 'off' };
  const pending = Math.max(0, input.pending);
  const failed = Math.max(0, input.failed);
  const left = pending + failed;
  if (left === 0) return { kind: 'all_sent' };
  if (!input.online) return { kind: 'offline', left };
  return { kind: 'sending', left, failed };
}

/** The status line under the thanks: finished, stopped early, or no session. */
export function doneStatusText(session: SessionRecord | null, t: Strings['onboarding']['done']): string {
  if (!session) return t.noSession;
  switch (session.status) {
    case 'ended_early':
      return t.endedEarly;
    case 'abandoned':
      return t.abandoned;
    default:
      return t.completed;
  }
}

function isOnline(): boolean {
  try {
    return typeof navigator === 'undefined' || navigator.onLine !== false;
  } catch {
    return true;
  }
}

export function DoneScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.done;
  const session = st.session;
  const [uploading, setUploading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [online, setOnline] = useState(isOnline);
  const prepared = useRef<{ key: string; bundle: ExportBundle } | null>(null);
  const bundleKey = session ? `${session.session_id}@${session.updated_at}` : '';

  let upload: UploadConfig | null = null;
  try {
    upload = effectiveUpload(st.config, st.settings.upload_override);
  } catch {
    upload = null;
  }
  const configured = uploadConfigured(upload);

  // the connection state is shown live (the upload queue itself retries on 'online')
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') return;
    const update = () => setOnline(isOnline());
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
    };
  }, []);

  useEffect(() => {
    let alive = true;
    const { store, uploader } = getServices();
    store.outboxCounts().then(
      (outbox) => {
        if (alive) setState({ outbox });
      },
      () => undefined,
    );
    // send what is still waiting right away (calls coalesce with a run already in flight)
    uploader.flush().then(
      (r) => {
        if (alive) setState({ outbox: r.counts });
      },
      () => undefined,
    );
    if (session) {
      buildBundle(session.session_id).then(
        (bundle) => {
          if (alive) prepared.current = { key: bundleKey, bundle };
        },
        (e) => console.warn('export bundle could not be prepared in advance', e),
      );
    }
    return () => {
      alive = false;
    };
  }, [bundleKey]);

  const onExport = () => {
    if (!session || exporting) return;
    const ready = prepared.current?.key === bundleKey ? prepared.current.bundle : null;
    setExporting(true);
    // with a prepared bundle, deliverExport (and navigator.share) runs inside this tap
    const delivery = ready ? deliverExport(ready) : buildBundle(session.session_id).then(deliverExport);
    delivery
      .then(
        (result) => showToast(t.exported[result], result === 'cancelled' ? 'info' : 'ok'),
        (e) => {
          console.error('export failed', e);
          showToast(t.exportFailed, 'error');
        },
      )
      .finally(() => setExporting(false));
  };

  const onUploadNow = () => {
    if (uploading) return;
    setUploading(true);
    getServices()
      .uploader.flush({ force: true })
      .then(
        (report) => {
          setState({ outbox: report.counts });
          if (report.attempted === 0) showToast(t.nothingToUpload, 'info');
          else showToast(t.uploadResult(report.sent, report.failed), report.failed > 0 ? 'warn' : 'ok');
        },
        (e) => {
          console.error('upload failed', e);
          showToast(s.common.errorGeneric, 'error');
        },
      )
      .finally(() => setUploading(false));
  };

  const goHome = () => {
    setState({ session: null });
    navigate('welcome');
  };

  const onNextPerson = () => {
    forgetParticipant();
    navigate('welcome');
  };

  const participant = st.participant;
  const name = firstName(participant?.nickname);
  const code = participant?.participant_id ?? session?.participant_id ?? null;
  const state = doneUploadState({ configured, pending: st.outbox.pending, failed: st.outbox.failed, online });
  const minutes = session ? sessionMinutes(session) : null;

  return (
    <Screen
      title={s.onboarding.welcome.title}
      testId="screen-done"
      footer={
        <>
          <Button size="lg" block testId="btn-next-person" onClick={onNextPerson}>
            {t.nextPerson}
          </Button>
          <Button variant="ghost" block testId="btn-home" onClick={goHome}>
            {t.home}
          </Button>
        </>
      }
    >
      <div class="onb-done-hero">
        <h2 class="onb-done-thanks" data-testid="done-thanks">
          {t.thanks(name)}
        </h2>
        <p>{doneStatusText(session, t)}</p>
        {session && (
          <p class="onb-done-count" data-testid="stat-accepted">
            {t.recorded(session.progress.n_accepted)}
          </p>
        )}
      </div>

      <Card title={t.uploadTitle} testId="upload-status">
        <div class="stack-sm" data-upload={state.kind}>
          {state.kind === 'off' && <p>{t.notConfigured}</p>}
          {state.kind === 'all_sent' && <Notice kind="ok">{t.allSent}</Notice>}
          {state.kind === 'sending' && <Notice kind="info">{t.sending(state.left)}</Notice>}
          {state.kind === 'sending' && state.failed > 0 && <p class="small muted">{t.failed(state.failed)}</p>}
          {state.kind === 'offline' && <Notice kind="warn">{t.offline(state.left)}</Notice>}
          {(state.kind === 'sending' || state.kind === 'offline') && (
            <Button variant="secondary" block disabled={uploading} testId="btn-upload-now" onClick={onUploadNow}>
              {uploading ? t.uploading : t.uploadNow}
            </Button>
          )}
        </div>
      </Card>

      {code && (
        <p class="onb-done-code" data-testid="done-code">
          {t.codeLabel} <bdi class="mono">{code}</bdi> <span class="muted">{t.codeHint}</span>
        </p>
      )}

      {session && (
        <Card title={t.detailsTitle} testId="done-summary">
          <Stat label={t.attempts} value={session.progress.n_attempts} testId="stat-attempts" />
          <Stat label={t.duration} value={minutes !== null ? s.common.minutes(Math.max(1, Math.round(minutes))) : '–'} />
          <div style={{ marginTop: '12px' }}>
            <Button variant="secondary" block disabled={exporting} testId="btn-export" onClick={onExport}>
              {exporting ? t.exporting : t.exportData}
            </Button>
          </div>
        </Card>
      )}
    </Screen>
  );
}
