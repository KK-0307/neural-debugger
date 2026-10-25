# Frontend

Next.js (App Router) + React + TypeScript + Tailwind. See the root README for how to run it.

```bash
npm run dev        # http://localhost:3000
npm test           # vitest (uses a real local HTTP server, no fetch stubs)
npm run lint && npm run typecheck && npm run build
```

`NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) is the backend URL, inlined at build time.
