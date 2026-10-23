/**
 * Hands an ExportBundle to the participant as a file. MUST be called from a tap (Web Share needs
 * transient activation): build the Blob/File synchronously-ish BEFORE anything slow if possible.
 * deliverExport() itself awaits nothing before navigator.share(); callers should load the bundle
 * (LocalStore.exportBundle, an IndexedDB read) BEFORE the tap — e.g. when the screen opens — because
 * WebKit does not carry the gesture across an IndexedDB await and share() then fails.
 * 1. JSON file (application/json, name …json): if navigator.canShare?.({files:[file]}) →
 *    navigator.share({files:[file], title}) → 'shared'.
 * 2. Chrome Android's share allowlist rejects .json/application/json: else try the SAME JSON content
 *    as text/plain with the name …json.txt; if canShare accepts it → share → 'shared'.
 *    (AbortError at any share step → 'cancelled'; other errors → fall through to download.)
 *    Files over 45 MB skip sharing (Chrome caps a share at 50 MB).
 * 3. Else download via an <a download> element with an object URL (revoked later) → 'downloaded'.
 *    Throws only when even that is impossible (no DOM).
 * File name: `nosetouch_${participant_id or 'all'}_${YYYYMMDD-HHMM}.json` (local time).
 *
 * IMPLEMENTED BY: data agent.
 */
import type { ExportBundle } from './types';

/** Chrome Android rejects shares over 50 MB; keep a margin. */
export const SHARE_MAX_BYTES = 45 * 1024 * 1024;
/** iOS needs the object URL alive while its download sheet is open. */
export const REVOKE_DELAY_MS = 60_000;
export const SHARE_TITLE = 'NoseTouch data export';

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

export function exportFileName(bundle: ExportBundle, now: Date = new Date()): string {
  const ids = new Set<string>();
  for (const p of bundle.participants ?? []) ids.add(p.participant_id);
  for (const s of bundle.sessions ?? []) ids.add(s.participant_id);
  for (const t of bundle.trials ?? []) ids.add(t.participant_id);
  const only = ids.size === 1 ? [...ids][0] : null;
  // ids are schema-checked, but a file name must never carry path separators or odd characters
  const who = only ? only.replace(/[^A-Za-z0-9_-]/g, '_') : 'all';
  const d = Number.isFinite(now.getTime()) ? now : new Date();
  const stamp =
    `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}` + `-${pad2(d.getHours())}${pad2(d.getMinutes())}`;
  return `nosetouch_${who}_${stamp}.json`;
}

interface ShareNavigator {
  canShare?: (data: ShareData) => boolean;
  share?: (data: ShareData) => Promise<void>;
}

function isAbortError(e: unknown): boolean {
  return !!e && typeof e === 'object' && (e as { name?: unknown }).name === 'AbortError';
}

function makeFile(content: Blob, name: string, type: string): File | null {
  try {
    return new File([content], name, { type });
  } catch {
    return null; // very old WebViews without the File constructor
  }
}

function canShareFile(nav: ShareNavigator, file: File): boolean {
  try {
    return nav.canShare?.({ files: [file] }) === true;
  } catch {
    return false; // some implementations throw instead of returning false
  }
}

function download(blob: Blob, name: string): void {
  const doc = (globalThis as { document?: Document }).document;
  if (!doc || typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
    throw new Error('This browser can neither share nor download files');
  }
  const url = URL.createObjectURL(blob);
  const a = doc.createElement('a');
  a.href = url;
  a.download = name;
  a.rel = 'noopener';
  a.style.display = 'none';
  doc.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.parentNode?.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), REVOKE_DELAY_MS);
  }
}

export async function deliverExport(bundle: ExportBundle): Promise<'shared' | 'downloaded' | 'cancelled'> {
  // Everything up to navigator.share() is synchronous: the tap's activation must still be alive.
  const name = exportFileName(bundle);
  const blob = new Blob([JSON.stringify(bundle)], { type: 'application/json' });
  const nav = (globalThis as { navigator?: ShareNavigator }).navigator;

  if (nav && typeof nav.share === 'function' && typeof nav.canShare === 'function' && blob.size <= SHARE_MAX_BYTES) {
    const candidates: [string, string][] = [
      [name, 'application/json'],
      [`${name}.txt`, 'text/plain'], // Chrome Android's allowlist accepts text/plain, not JSON
    ];
    for (const [fileName, type] of candidates) {
      const file = makeFile(blob, fileName, type);
      if (!file || !canShareFile(nav, file)) continue;
      try {
        await nav.share({ files: [file], title: SHARE_TITLE });
        return 'shared';
      } catch (e) {
        if (isAbortError(e)) return 'cancelled';
        break; // e.g. NotAllowedError (activation expired): a second share would fail too
      }
    }
  }

  download(blob, name);
  return 'downloaded';
}
