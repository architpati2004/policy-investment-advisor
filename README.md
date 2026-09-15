# Policy-Based Investment Advisor (RAG)

A fully local, retrieval-augmented research assistant that reads government and
regulatory policy (RBI circulars, SEBI regulations, Budget documents) and company
filings, and answers questions about them **against a specific portfolio** — with
citations, and with the ability to say that the sources do not answer the question.

Everything runs offline on an Apple Silicon MacBook Air using **Ollama** for
inference and **FAISS** for vector search. No OpenAI key, no Anthropic key, no paid
vector database, no paid news API.

> An educational research tool. Answers are grounded in the documents you index and
> cite the chunks they came from; they are not financial advice. The portfolio
> tracks **cost basis only** — there is no price feed, so nothing here is a
> valuation.

Two properties matter more than feature count, and both are tested:

- **It refuses.** Asked something the indexed documents cannot answer — NBFC deposit
  rules when the corpus holds only commercial-bank rules — it replies `NOT_COVERED`
  rather than applying one set of rules to an institution they were not written for.
- **It verifies its own output.** A citation pointing at a source that was never
  supplied is recorded as invalid rather than resolved to some other chunk, and an
  answer that cites nothing is reported as ungrounded.

**[CLAUDE.md](CLAUDE.md) is the companion document**: it records why each design
decision was made, the measurements behind them, and — in section 11 — the known
defects that are still unfixed. Read it before trusting any number this system
produces.

---

## Status

Phases 1-8 are complete. Phases 9-14 are not started.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Project structure + environment | ✅ done |
| 2 | Local Ollama connection | ✅ done |
| 3 | PDF ingestion | ✅ done |
| 4 | Embeddings + policy FAISS index | ✅ done |
| 5 | Policy RAG | ✅ done |
| 6 | Company FAISS index | ✅ done |
| 7 | Portfolio database | ✅ done |
| 8 | Portfolio-aware RAG | ✅ done |
| 9 | News ingestion (RSS) | pending |
| 10 | Alert engine | pending |
| 11 | FastAPI endpoints | pending |
| 12 | React frontend | pending |
| 13 | Tests | pending |
| 14 | README + demo | pending |

Test suite: **234 tests** in the default run (~25 s), plus 4 marked `slow` that run
real local generation (`pytest -m slow`, ~2 minutes). Live tests skip themselves
when Ollama is not running.

---

## Stack

| Layer | Choice |
| --- | --- |
| LLM | Ollama + Qwen — `qwen3:1.7b` for development, `qwen3:4b` for quality |
| Embeddings | Ollama + `embeddinggemma` (768 dimensions) |
| Vector store | FAISS — two independent indexes (policy, company), cosine similarity |
| Orchestration | LangChain (`langchain-core`, `langchain-community`, `langchain-ollama`) |
| Ingestion | pypdf, custom cleaning and chunking |
| Database | SQLite + SQLAlchemy 2.0 |
| Backend | FastAPI + Pydantic (system endpoints only until Phase 11) |
| Frontend | React + Vite (Phase 12) |

---

## Setup

### 1. Python and dependencies

```bash
cd policy-investment-advisor

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Shortcuts: `make setup`, `make env`. Python 3.11 or 3.12.

### 2. Ollama

Install from <https://ollama.com> and leave the server running:

```bash
ollama serve
```

On macOS 14 the Homebrew formula wants to build from source; the packaged
installer from ollama.com is the working route.

### 3. Models

```bash
make models
```

which is the three pulls, one model per invocation:

```bash
ollama pull qwen3:1.7b      # ~1.4 GB, the development model
ollama pull qwen3:4b        # ~2.5 GB, for quality checks and demos
ollama pull embeddinggemma  # ~0.6 GB, embeddings for both indexes
```

`.env` ships with `qwen3:1.7b`, which answers in roughly 25 s against ~125 s for
`qwen3:4b`. Develop on the small one; switch to `qwen3:4b` — and raise
`LLM_TIMEOUT_SECONDS` to `300` — before judging answer quality or recording a demo.
CLAUDE.md §11 explains what the small model costs you (looser citations) and why
prompt changes must always be checked on 4b.

### 4. Verify

```bash
python scripts/check_setup.py     # environment and imports
python scripts/check_ollama.py    # server, models, one timed generation
pytest                            # 234 tests
```

---

## Using it

The three CLIs below build the two indexes and then ask questions against them.
Each has `--help`.

### Build the policy index

Put RBI/SEBI/Budget PDFs in `data/policies/`. A filename beginning with a known
publisher (`RBI_…`, `SEBI_…`) is attributed automatically; otherwise pass
`--source`.

```bash
python scripts/build_policy_index.py build          # or: make index
python scripts/build_policy_index.py build --rebuild
python scripts/build_policy_index.py query "can a bank pay interest on a current account?"
python scripts/build_policy_index.py stats          # or: make index-stats
```

Re-running only embeds what is new — chunk ids are content-hashed, so an unchanged
document is skipped rather than duplicated. Four RBI circulars (~30 pages) produce
about 73 chunks in a few seconds.

To ask a policy question and get a written, cited answer rather than raw chunks:

```bash
python scripts/ask_policy.py "what are the interest rate rules for deposits at commercial banks?"
```

### Build the company index

Declare each company once in `data/companies/registry.json`, then drop its filings
in `data/companies/`:

```json
[
  {
    "ticker": "GODREJCP",
    "name": "Godrej Consumer Products",
    "sector": "FMCG",
    "aliases": ["godrej consumer", "gcpl"]
  }
]
```

```bash
python scripts/build_company_index.py build
python scripts/build_company_index.py query "what drove revenue growth?" --company GODREJCP
python scripts/build_company_index.py query "input cost inflation" --sector FMCG
python scripts/build_company_index.py companies     # declared vs indexed
```

The registry resolves whatever filename a filing arrives with — `godrej consumer.pdf`,
`GODREJCP_AR_2025.pdf` and `gcpl.pdf` all match the same company. A file that cannot
be attributed is **refused**, not indexed: a chunk with no company can never be
matched to a holding, but could still surface in a search and be cited. A 589-page
annual report yields ~1,776 chunks in about 3 minutes.

### Assess a question against the portfolio

Create a portfolio first — `assess` needs one:

```bash
python scripts/portfolio.py init        # create tables, import the registry
python scripts/portfolio.py seed-demo   # illustrative 150 GODREJCP at 1180.50
python scripts/portfolio.py add --ticker GODREJCP --quantity 150 --cost 1180.50
python scripts/portfolio.py list        # holdings, cost basis, weights
python scripts/portfolio.py scope       # the companies and sectors retrieval uses
```

Then:

```bash
python scripts/assess.py "do the new commercial bank deposit rate rules affect my holdings?"
python scripts/assess.py "what regulatory risks affect consumer goods?" --show-context
```

`assess` derives the companies and sectors you hold, searches the policy index
(unscoped) and the company index (narrowed to your holdings), merges the two under
a quota so neither can crowd the other out, and reports which holdings the sources
show are affected — and which are not. Expect ~25 s on `qwen3:1.7b`.

### Run the API

```bash
uvicorn backend.main:app --reload     # or: make run
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/system/ollama
```

System endpoints only for now: health, configuration, and local-inference status
(200 ready, 503 server down, 424 model not pulled). Feature routes arrive in
Phase 11.

---

## Configuration

Everything lives in `.env`, loaded by `backend/config.py` into a typed `Settings`
object. No model name, URL or path is hard-coded anywhere else.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `LLM_MODEL` | `qwen3:1.7b` | Generation model (`qwen3:4b` for quality) |
| `EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `LLM_TEMPERATURE` | `0.1` | Low, to reduce fabrication |
| `LLM_TIMEOUT_SECONDS` | `180` | Wall clock on a generation; use `300` for 4b |
| `LLM_NUM_CTX` | `8192` | Context window |
| `EMBEDDING_BATCH_SIZE` | `16` | Texts per embedding request |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text splitting |
| `RETRIEVAL_TOP_K` | `5` | Chunks per retriever |
| `MIN_RELEVANCE_SCORE` | `0.35` | Absolute floor: below it, the corpus is not about the question |
| `RELATIVE_RELEVANCE_RATIO` | `0.75` | Keep chunks within this fraction of the best match |
| `DATA_DIR` | `data` | Source documents |
| `VECTORSTORE_DIR` | `backend/vectorstore` | FAISS indexes |
| `DATABASE_URL` | `sqlite:///data/app.db` | SQLite database |

The two relevance floors do different jobs, and the second is what scales as a
corpus grows. CLAUDE.md §11 has the measurements, including why **no** absolute
score can separate a question the corpus answers from a near-identical one it
cannot.

---

## Project layout

```
policy-investment-advisor/
├── CLAUDE.md                 # design decisions, measurements, known defects
├── backend/
│   ├── config.py             # typed settings from .env
│   ├── exceptions.py         # error hierarchy, each with a remediation string
│   ├── main.py               # FastAPI factory
│   ├── api/system.py         # /health, /api/config, /api/system/ollama
│   ├── ingestion/            # PDF loading, cleaning, chunking, company registry
│   ├── rag/
│   │   ├── ollama_client.py  # connection, model checks, generation deadline
│   │   ├── embeddings.py     # batched, unit-normalised embeddings
│   │   ├── vector_store.py   # FAISS persistence, dedupe, scored search
│   │   ├── retrieval.py      # relevance floors, scoping, context merging
│   │   ├── prompts.py        # grounding prompts and refusal sentinels
│   │   ├── policy_rag.py     # Phase 5 chain
│   │   └── portfolio_rag.py  # Phase 8 chain
│   ├── db/                   # SQLAlchemy models, sessions, portfolio operations
│   ├── alerts/               # Phase 10
│   └── vectorstore/          # FAISS indexes (gitignored)
├── data/                     # source documents (gitignored except the registry)
├── scripts/                  # the CLIs described above
└── tests/                    # one module per phase
```

Boundaries kept throughout: LangChain imports stay in `backend/rag/`, ingestion
never imports retrieval, and SQLAlchemy models stay out of route modules.

---

## Make targets

```
make setup     # venv + dependencies        make index       # build the policy index
make env       # .env from the example      make index-stats # describe it
make models    # pull the three models      make test        # pytest
make run       # start the dev server       make verify      # setup checker
make clean     # remove venv and caches     make verify-ollama
```

---

## Known limitations

Recorded in full, with evidence, in [CLAUDE.md](CLAUDE.md) §11. The ones that would
mislead you first:

- **Financial figures cannot be trusted yet.** Table extraction drops column
  headers, so a chunk reads `Net profit margin (%) 7.83% 20.10` — two years with
  nothing saying which is which.
- **Marketing prose outranks disclosure** in the company index; "what drove revenue
  growth?" returns aspirational filler above the actual figures.
- **No-impact answers cite nothing.** When the conclusion is that a holding is
  unaffected, the answer comes back ungrounded on both models — and those are
  precisely the assessments Phase 10 will build alerts on.
- **The two models disagree** about whether "no impact" is an answer or a refusal,
  with the larger one more conservative. Read the reason, not just the label.
