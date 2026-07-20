import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// The API returns relative URLs (thumbnail_url is "/thumbnails/{id}"), so the
// app never needs to know where the backend lives. In development that only
// works if the dev server forwards those paths; in production FastAPI serves
// the built files itself and they are same-origin already.
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    proxy: {
      "/search": API,
      "/videos": API,
      "/thumbnails": API,
      "/health": API,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
