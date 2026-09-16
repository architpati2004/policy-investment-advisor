# Frontend

React + Vite + TypeScript. Talks to the FastAPI backend on port 8000 through a
dev-server proxy, so the browser sees one origin and CORS never enters the
picture.

```bash
npm install
npm run dev        # http://localhost:5173
npm run typecheck
npm run build
```

The backend must be running:

```bash
cd .. && source .venv/bin/activate && uvicorn backend.main:app --reload
```

## Layout

```
src/
├── services/     every network call — nothing else imports fetch
├── components/   GenerationProgress, ResultCard, SourceList, PortfolioContext
├── pages/        Chat, Dashboard, Alerts, Portfolio, Documents
└── types.ts      API contracts, mirroring backend/api/schemas.py
```

Two things about this UI are deliberate and worth not "fixing":

**Generation shows an elapsed counter, not a spinner.** A local answer takes
30-90 seconds. A spinner that has looked identical for forty seconds is
indistinguishable from a hung request, and the user's only recourse is to reload
and lose the work. The counter ticking is proof of life; the stage text says what
is happening and how long it should take. The bar is indeterminate on purpose —
an invented percentage stalling at 80% is worse than none.

**A refusal is shown as a finding, not an error.** `covered: false` means the
indexed documents do not answer the question, and the explanation usually
contains the useful part ("the sources govern commercial banks, so this does not
reach an FMCG holding"). It renders as a neutral card — never red, no retry
prompt — because nothing failed. Equally it is never dressed up as an answer: the
heading says the corpus does not cover this, so the reader knows what they have.
