/**
 * Entry of the predict chunk (main.tsx loads it with a dynamic import only on ?predict=1 or
 * ?mode=predict, so the collector's bundle, storage and behaviour are untouched).
 *
 * mount(): parses PredictParams, blocks every network connection for the life of the page with a
 * Content-Security-Policy meta (connect-src 'none': fetch, XHR, WebSocket and sendBeacon; production
 * builds only, the dev server's HMR socket needs it), sets lang/dir on <html> and renders PredictApp.
 * No config.json fetch, no IndexedDB, no uploader, no service worker registration.
 */
import { render } from 'preact';
import { PredictApp, type PreviewState } from './PredictApp';
import { binding } from './live/engine';
import { parsePredictParams } from './ui/params';
import './styles.css';

export const CSP_CONTENT = "connect-src 'none'";

/** Adds the connect-src 'none' policy once (it cannot be lifted again for this page). */
export function installNoNetworkPolicy(doc: Document): HTMLMetaElement {
  const existing = doc.head.querySelector<HTMLMetaElement>('meta[http-equiv="Content-Security-Policy"][data-predict]');
  if (existing) return existing;
  const meta = doc.createElement('meta');
  meta.httpEquiv = 'Content-Security-Policy';
  meta.content = CSP_CONTENT;
  meta.setAttribute('data-predict', '1');
  doc.head.appendChild(meta);
  return meta;
}

export function mount(root: HTMLElement, search: string): () => void {
  const params = parsePredictParams(search);
  const doc = root.ownerDocument;
  if (import.meta.env.PROD) installNoNetworkPolicy(doc);
  doc.documentElement.lang = params.lang;
  doc.documentElement.dir = params.lang === 'he' ? 'rtl' : 'ltr';
  doc.title = params.lang === 'he' ? 'NoseTouch · באיזו יד?' : 'NoseTouch · Which hand?';

  let disposed = false;
  const show = (preview: PreviewState | null): void => {
    if (!disposed) render(<PredictApp params={params} binding={binding} preview={preview} />, root);
  };
  const previewName = import.meta.env.DEV ? new URLSearchParams(search).get('preview') : null;
  if (import.meta.env.DEV && previewName) {
    // dev-only fixed states for screenshots (never in a production build)
    void import('./ui/preview').then((m) => show(m.previewState(previewName, binding.models(), params.model)));
  } else {
    show(null);
  }
  return () => {
    disposed = true;
    render(null, root);
  };
}
