import { execSync } from 'node:child_process';
import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

/** the short commit of the build (GitHub Actions: GITHUB_SHA), shown in the predict page's footer; '' without git */
function buildId(): string {
  if (process.env.GITHUB_SHA) return process.env.GITHUB_SHA.slice(0, 7);
  try {
    return execSync('git rev-parse --short=7 HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return '';
  }
}

// Testing on a real phone needs HTTPS (motion sensors require a secure context). Preferred:
//   npm run dev -- --host   +   cloudflared tunnel --url http://localhost:5173   (no account needed;
//   gives a real certificate, so permissions and the service worker behave like production).
// Fallback: `npm run dev:phone` (mode "phone") = self-signed HTTPS on the LAN; Chrome then refuses to
// register service workers and iOS behaviour after the certificate warning is undocumented.
// Uploads to a local receiver go through the /api proxy (server/receiver.py) so the phone never
// talks plain HTTP.
const tunnelHosts = ['.trycloudflare.com'];

export default defineConfig(({ mode }) => ({
  base: './', // relative base: works on GitHub Pages sub-paths and any static host
  plugins: [preact(), ...(mode === 'phone' ? [basicSsl()] : [])],
  define: { __PREDICT_BUILD__: JSON.stringify(mode === 'test' ? '' : buildId()) },
  server: {
    port: 5173,
    allowedHosts: tunnelHosts, // Vite ≥ 6.0.9 blocks unknown Host headers (CVE-2025-24010)
    proxy: {
      '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
    },
  },
  preview: {
    allowedHosts: tunnelHosts,
  },
  build: {
    target: 'es2020', // iOS Safari 14+ / Chrome 87+
    sourcemap: true,
    // the predict chunk (?predict=1, loaded on demand) carries the final CNN's float64 weights (about 1.1 MB,
    // 0.47 MB gzip); the collector's own entry chunk stays far below this
    chunkSizeWarningLimit: 1200,
  },
  test: {
    // tests/: the collector; src/predict/: predict mode's tests next to its code (the real-recording fixtures they
    // read from analysis/data/stage3_fixtures are git-ignored, so those blocks skip in CI)
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts', 'src/**/*.test.tsx'],
    environment: 'node',
    testTimeout: 20000,
  },
}));
