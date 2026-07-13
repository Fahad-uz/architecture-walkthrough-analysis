import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API calls to the FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/jobs": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/gemini-status": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1600,
  },
});
