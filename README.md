# Policy-Based Investment Advisor

A retrieval-augmented research assistant that reads financial regulation, company
filings and news against a specific portfolio, and answers questions about them
with citations — or says the documents do not answer the question.

It runs entirely on a MacBook Air M1: **Ollama** for generation and embeddings,
**FAISS** for vector search, SQLite for the portfolio. No OpenAI key, no
Anthropic key, no paid vector database, no paid news API, no network calls at all
except fetching the RSS feeds you configure.

> Educational research tool. Answers are grounded in whatever you have indexed
> and cite the chunks they came from. The portfolio tracks **cost basis only** —
> there is no price feed, so nothing here is a valuation. Not investment advice.

## What it does, and what it refuses to do

Two behaviours matter more than the feature list, and both are tested:

**It refuses.** Asked what deposit rates an NBFC may offer when the corpus holds
only commercial-bank rules, it answers `NOT_COVERED: the sources govern
commercial banks, not NBFCs` rather than applying one institution's rules to
another. This case is hard precisely because retrieval looks confident: those
bank chunks score **0.539**, higher than three questions the corpus genuinely
answers. No relevance threshold can separate them, so the judgement is asked of
the model and verified in code.

**It verifies its own output.** A citation pointing at a source that was never
supplied is recorded as invalid rather than resolved to some other chunk. An
answer citing nothing is reported as ungrounded. An alert whose only evidence is
the company's own annual report is discarded, because nothing in a filing is new.

---

## Setup

### 1. Python and dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env
```

Python 3.11 or 3.12. Shortcuts: `make setup`, `make env`.

### 2. Ollama

Install from <https://ollama.com> — on macOS 14 the Homebrew formula wants to
build from source — then leave the server running:

```bash
ollama serve
```

### 3. Models

```bash
make models      # three pulls, one model per invocation
```

| model | size | role |
| --- | --- | --- |
| `qwen3:1.7b` | 1.4 GB | development and alerts (~25 s per answer) |
| `qwen3:4b` | 2.5 GB | quality checks and demos (~125 s per answer) |
| `embeddinggemma` | 0.6 GB | embeddings for all three indexes, 768 dimensions |

`.env` ships with 1.7b. Switch `LLM_MODEL` to `qwen3:4b` — and raise
`LLM_TIMEOUT_SECONDS` to `300` — before judging answer quality. See
[CLAUDE.md](CLAUDE.md) §11 for what the small model costs you.

### 4. Verify

```bash
python scripts/check_setup.py     # environment and imports
python scripts/check_ollama.py    # server, models, one timed generation
pytest                            # 353 tests, ~30 s
```

---

## Using it

### Index your documents

Drop RBI/SEBI/Budget PDFs in `data/policies/`, company filings in
`data/companies/`, and declare each company once in
`data/companies/registry.json`:

```json
[{ "ticker": "GODREJCP", "name": "Godrej Consumer Products",
   "sector": "FMCG", "aliases": ["godrej consumer", "gcpl"] }]
```

```bash
python scripts/build_policy_index.py build      # or: make index
python scripts/build_company_index.py build
python scripts/build_news_index.py build        # RSS feeds from data/news/feeds.json
```

Re-running only embeds what is new: chunk ids are content-hashed, so an unchanged
document is skipped rather than duplicated. Roughly: four RBI circulars → 73
chunks in seconds; a 589-page annual report → 1,776 chunks in about 3 minutes.

### Build a portfolio

```bash
python scripts/portfolio.py init        # tables + import the registry
python scripts/portfolio.py seed-demo   # illustrative 150 GODREJCP at ₹1180.50
python scripts/portfolio.py list
python scripts/portfolio.py scope       # what retrieval narrows to
```

### Ask

```bash
python scripts/ask_policy.py "can a bank pay interest on a current account?"
python scripts/assess.py "do the new deposit rate rules affect my holdings?"
python scripts/alerts.py run            # one generation per holding
```

### Run the app

```bash
uvicorn backend.main:app --reload      # terminal 1 — API on :8000, docs at /docs
cd frontend && npm install && npm run dev   # terminal 2 — UI on :5173
```

Vite binds IPv6, so use `localhost:5173` rather than `127.0.0.1:5173`.

---

## Architecture

```
                    ┌──────────────┐
   portfolio ──────▶│ companies +  │
   (SQLite)         │ sectors      │
                    └──────┬───────┘
                           │ scope
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  policy_index       company_index       news_index      three FAISS indexes,
  (regulation)       (filings)           (RSS)           cosine similarity
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                   relevance floors          absolute + corpus-scaling
                           ▼
                    quota merge              each corpus guaranteed a share
                           ▼
                  grounding prompt           cites, or refuses
                           ▼
                   qwen3 via Ollama
                           ▼
        answer + verified citations + impact assessment
```

```
backend/
├── config.py          typed settings; nothing hard-codes a model or path
├── exceptions.py      every error carries the command that fixes it
├── ingestion/         PDF loading, cleaning, chunking, company registry, RSS
├── rag/               embeddings, FAISS, retrieval policy, prompts, two chains
├── db/                SQLAlchemy models, sessions, portfolio, alerts
├── alerts/            the alert engine
└── api/               FastAPI routers, schemas, dependencies
frontend/src/
├── services/          every network call; nothing else imports fetch
├── components/        progress, result card, sources, portfolio context
└── pages/             chat, dashboard, alerts, portfolio, documents
```

---

## Design decisions

Each of these was measured, and each is recorded with its evidence in
[CLAUDE.md](CLAUDE.md) §11.

**Two relevance floors, not one.** An absolute floor (0.35) rejects questions the
corpus is not about; a relative floor (0.75 of the best match) drops the tail and
is the part that scales, because the bar rises as the corpus improves. Neither
can catch the adjacent case — that is the prompt's job.

**Three indexes, not one.** Merging corpora by raw score is winner-take-all: on
real questions it produced 5-0 and 0-5 splits, one corpus shutting the other out
entirely. Each is guaranteed a share of the context instead, because an impact
assessment needs the rule and the holding in front of the model at once.

**Recency applies to news only.** A circular that has not been amended binds as
much today as when it was published; an article is a claim about a moment. News
decays with a 45-day half-life and a floor, and decay reorders — never filters —
so an old article that clears the relevance floor still appears.

**Money is stored as integers.** Paise and thousandths of a share. SQLite has no
decimal type, so a `Numeric` column round-trips through a float, and a portfolio
multiplies and sums these constantly.

**Alerts iterate holdings, not documents.** Company attribution on live RSS runs
at about 2.5%, so an engine triggered by "an article names your company" would be
silent for reasons unrelated to whether anything happened. Cost is bounded by
portfolio size: one holding, one local generation.

**Prompt wording dominates local latency.** A five-rule prompt generated 2,237
tokens where a four-line one generated 259 — 265 s against 45 s, same question,
same context. Halving the context changed nothing.

**A refusal is a result, not an error.** The API returns 200 with the finding in
the body and the UI renders it as a neutral card, because the refusal text
routinely *is* the answer: "the sources govern commercial banks, so this does not
reach an FMCG holding."

---

## Known limitations

The full list, with evidence, is in [CLAUDE.md](CLAUDE.md) §11. The ones that
would mislead you first:

- **Financial figures cannot be trusted.** Table extraction drops column headers,
  so a chunk reads `Net profit margin (%) 7.83% 20.10` — two years, nothing
  saying which is which. A model citing it will state the wrong year confidently.
- **Marketing prose outranks disclosure** in the company index: "what drove
  revenue growth?" returns aspirational filler above the actual figures.
- **No-impact answers cite nothing** on both models, so the assessments alerts are
  built from are the least verifiable the system produces.
- **The two models disagree** about whether "no impact" is an answer or a refusal,
  with the larger one more conservative — read the reason, not the label. And
  1.7b is the likelier of the two to manufacture an applicability link.
- **`qwen3:4b` cannot write an alert summary** within a workable deadline: terse
  enough to finish and the output is empty, verbose enough to explain and it
  times out. Alerts run on 1.7b for this reason.
- **News attribution is ~2.5%** on general market feeds. Per-company feeds would
  raise it; the engine deliberately does not depend on it.

---

## Development

```
make test          # 353 backend tests, ~30 s
make test-slow     # 4 live-generation tests, ~2 minutes
make coverage      # 96% of backend statements
make frontend-test # 18 frontend tests
```

Backend tests need neither Ollama nor a network: a deterministic in-process
embedding model means the FAISS indexes under test are real indexes. Tests that
do need a live model skip themselves when one is not running.

Conventions worth keeping are in [CLAUDE.md](CLAUDE.md) §10 — typed settings,
LangChain confined to `backend/rag/`, ingestion separate from retrieval, database
models out of routes, and modules imported *as modules* where tests need to
substitute them.
