import { render } from 'preact';
import { App } from './app/App';
import './ui/styles.css';

const root = document.getElementById('app')!;
const query = new URLSearchParams(location.search);

if (query.get('predict') === '1' || query.get('mode') === 'predict') {
  // Predict mode (Stage 3, docs/stage3/APP_SPEC.md): its own chunk, loaded only here. The collector is not
  // rendered, so there is no config.json fetch, no IndexedDB, no uploader and no service worker registration.
  // Production: no network connection at all (connect-src 'none') from before the chunk runs; mount() keeps it.
  if (import.meta.env.PROD) {
    const meta = document.createElement('meta');
    meta.httpEquiv = 'Content-Security-Policy';
    meta.content = "connect-src 'none'";
    meta.setAttribute('data-predict', '1');
    document.head.appendChild(meta);
  }
  root.textContent = 'Loading…'; // the chunk is about 0.5 MB gzip: a slow connection shows this, not a blank page
  const failed = (): void => {
    root.textContent = 'Could not load. Check the connection and reload.';
  };
  // the outer catch also covers a stylesheet that failed to preload (Vite's preload wrapper throws before .then)
  import('./predict/mount').then((m) => m.mount(root, location.search), failed).catch(failed);
} else {
  render(<App />, root);

  // Offline support (cached app shell). Only in production builds: the dev server must not be cached.
  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./sw.js').catch((e) => console.warn('service worker registration failed', e));
    });
  }
}
