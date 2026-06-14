import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The app is served by forgetfuldb-server at /ui, so assets resolve
// relative to that base. `npm run dev` proxies API calls to the local
// server so the SPA works unbuilt too.
export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    proxy: Object.fromEntries(
      ['/graph', '/retrieve', '/uiconfig', '/turns', '/consolidations', '/memory', '/stats', '/metrics'].map(
        (p) => [p, { target: 'http://127.0.0.1:8787', changeOrigin: false }],
      ),
    ),
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1500 },
});
