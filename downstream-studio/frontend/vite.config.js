import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(import.meta.dirname, '../src/downstream_studio/web'),
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: 'charts', test: /node_modules\/(recharts|d3-|victory-vendor)/ },
            { name: 'react', test: /node_modules\/(react|react-dom)/ },
          ],
        },
      },
    },
  },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
