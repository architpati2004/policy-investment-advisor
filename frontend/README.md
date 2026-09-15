# Frontend (Phase 12)

The React + Vite dashboard is scaffolded in Phase 12. The directory layout is
reserved now so the structure stays stable:

```
src/
  components/   # reusable UI (holding rows, alert cards, source lists)
  pages/        # Dashboard, Portfolio, AI Research Chat, Alerts, Documents
  services/     # ALL backend HTTP calls live here — nowhere else
  App.jsx
```

When Phase 12 starts, it will be created with:

```bash
npm create vite@latest frontend -- --template react
cd frontend && npm install && npm run dev
```

The dev server runs on http://localhost:5173, which is already allow-listed in
`CORS_ORIGINS` in `.env`.
