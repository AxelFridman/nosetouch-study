/**
 * End of a session: thanks, a short summary, upload status (state.outbox) with "Upload now", and a
 * "Save / share my data" export of this session.
 *
 * The export bundle is prepared as soon as the screen opens, so that the tap calls deliverExport
 * synchronously: iOS only opens the share sheet inside a user gesture, which an IndexedDB read
 * before navigator.share() would outlive. If the bundle is not ready yet, it is built on the tap.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type { ExportBundle, UploadConfig } from '../../core/types';
import { effectiveUpload, redactProfile } from '../../core/config';
import { deliverExport } from '../../core/exporter';
import { Button, Card, Notice, Screen, Stat } from '../../ui/components';
import { useStrings } from '../../i18n';
import { getServices } from '../services';
import { getState, navigate, setState, showToast, useAppState } from '../store';
import { sessionMinutes } from '../sessionFlow';
import '../../ui/session.css';

function buildBundle(sessionId: string): Promise<ExportBundle> {
  return getServices().store.exportBundle({
    sessionIds: [sessionId],
    redact: (p) => redactProfile(p, getState().config),
  });
}

function uploadConfigured(cfg: UploadConfig | null): boolean {
  return !!cfg && cfg.mode !== 'none' && cfg.endpoint.trim() !== '';
}

export function DoneScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.session.done;
  const session = st.session;
  const [uploading, setUploading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const prepared = useRef<{ key: string; bundle: ExportBundle } | null>(null);
  const bundleKey = session ? `${session.session_id}@${session.updated_at}` : '';

  let upload: UploadConfig | null = null;
  try {
    upload = effectiveUpload(st.config, st.settings.upload_override);
  } catch {
    upload = null;
  }
  const configured = uploadConfigured(upload);

  useEffect(() => {
    let alive = true;
    const { store } = getServices();
    store.outboxCounts().then(
      (outbox) => {
        if (alive) setState({ outbox });
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

  const { pending, failed } = st.outbox;
  const minutes = session ? sessionMinutes(session) : null;

  return (
    <Screen
      title={t.title}
      testId="screen-done"
      footer={
        <>
          {session && (
            <Button size="lg" block disabled={exporting} testId="btn-export" onClick={onExport}>
              {exporting ? t.exporting : t.exportData}
            </Button>
          )}
          <Button variant="secondary" block testId="btn-home" onClick={goHome}>
            {t.home}
          </Button>
        </>
      }
    >
      <p class="big">
        {!session ? t.noSession : session.status === 'abandoned' ? t.abandoned : t.completed}
      </p>

      {session && (
        <Card title={t.summary} testId="done-summary">
          <Stat
            label={t.accepted}
            value={`${session.progress.n_accepted} / ${session.protocol.n_planned_trials}`}
            testId="stat-accepted"
          />
          <Stat label={t.attempts} value={session.progress.n_attempts} testId="stat-attempts" />
          <Stat label={t.duration} value={minutes !== null ? s.common.minutes(Math.max(1, Math.round(minutes))) : '–'} />
        </Card>
      )}

      <Card title={t.upload} testId="upload-status">
        {!configured ? (
          <p>{t.notConfigured}</p>
        ) : pending + failed === 0 ? (
          <Notice kind="ok">{t.allSent}</Notice>
        ) : (
          <>
            {pending > 0 && <p>{t.pending(pending)}</p>}
            {failed > 0 && <Notice kind="warn">{t.failed(failed)}</Notice>}
            <p class="muted small">{t.keepOpen}</p>
          </>
        )}
        <div style={{ marginTop: '12px' }}>
          <Button
            variant="secondary"
            block
            disabled={!configured || uploading}
            testId="btn-upload-now"
            onClick={onUploadNow}
          >
            {uploading ? t.uploading : t.uploadNow}
          </Button>
        </div>
      </Card>
    </Screen>
  );
}
