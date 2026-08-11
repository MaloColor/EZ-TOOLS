import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // lets `npm run dev` hit local `vercel dev` for the /api functions if desired
      "/api": "http://localhost:3000",
    },
  },
});
