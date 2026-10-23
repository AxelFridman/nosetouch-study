import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

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
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 20000,
  },
}));
