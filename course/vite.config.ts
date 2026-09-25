/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from the root of the custom domain marqa.tech (see public/CNAME).
// Hash routing means every lesson URL is still a single static page.
export default defineConfig({
  base: '/',
  plugins: [react()],
  // src/data/repoFiles.ts imports the repo's Python scripts (../phase*/) with ?raw; the dev server
  // must be allowed to read that folder. The build inlines them and needs nothing extra.
  server: { fs: { allow: ['..'] } },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
