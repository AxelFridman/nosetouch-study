import { render } from 'preact';
import { App } from './app/App';
import './ui/styles.css';

render(<App />, document.getElementById('app')!);

// Offline support (cached app shell). Only in production builds: the dev server must not be cached.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch((e) => console.warn('service worker registration failed', e));
  });
}
