# CLAUDE.md — Project Context and Working Agreement

## Working agreement (read this first)

- **Build one phase at a time.** After completing a phase, stop and wait for my
  confirmation before starting the next. Do not run ahead.
- After each phase, tell me: what was implemented, which files changed, the exact
  commands to run it, and a small test command.
- **Never weaken a test to make it pass.** If a test fails, fix the code, or tell
  me the test was wrong and why. Do not delete assertions or loosen them silently.
- Run `pytest` before declaring a phase complete.
- Keep the code-quality rules in section 10 below.
- Everything must run locally on an M1 MacBook Air with no paid APIs.

---

## 1. Goal

A full-stack RAG application that contextualises government/regulatory policy,
financial news and company information against a user's investment portfolio,
and produces **grounded, cited** answers plus portfolio impact alerts.

Hard constraint: it must run on a **MacBook Air M1** using only local/free
inference. No OpenAI key, no Anthropic key, no paid embedding API, no paid
vector database, no paid news API.

## 2. Environment (verified working)

| Item | Value |
|---|---|
| Machine | MacBook Air M1, macOS 14 |
| Python | 3.12.14 (Homebrew), in a `.venv` at the project root |
| Inference | Ollama 0.34.0, installed from ollama.com (not Homebrew) |
| Generation model | `qwen3:4b` (~2.5 GB) |
| Embedding model | `embeddinggemma` (~621 MB) |
| Project path | `~/Downloads/policy-investment-advisor` |

Measured: first generation took **69.8 s** (cold model load plus Qwen3's
internal reasoning). Later calls are faster while the model stays resident.

## 3. Tech stack

**Backend:** Python, FastAPI, LangChain (`langchain`, `langchain-core`,
`langchain-community`, `langchain-ollama`, `langchain-text-splitters`), FAISS
(`faiss-cpu`, native arm64 wheel — no compilation on M1), pypdf, SQLAlchemy,
Pydantic + pydantic-settings, feedparser, BeautifulSoup, requests.

**Frontend (not built yet):** React + Vite.

**Storage:** SQLite via SQLAlchemy; two local FAISS indexes.

## 4. Architecture

Two independent FAISS indexes:

- **policy_index** — RBI/SEBI documents, budget PDFs, government policy,
  regulatory circulars.
- **company_index** — company fundamentals, annual reports, quarterly results,
  company and sector news.

Intended query flow:

```
user portfolio ──> derive companies + sectors
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  policy retriever                   company retriever
        └─────────────────┬─────────────────┘
                          ▼
                   context merger
                          ▼
                  LangChain RAG chain
                          ▼
                  local Qwen LLM (Ollama)
                          ▼
     grounded answer + sources + impact assessment
```

## 5. Development method

Built in 14 phases, one at a time, each verified before the next starts.

| Phase | Scope | Status |
|---|---|---|
| 1 | Project structure + environment | done |
| 2 | Local Ollama connection | done |
| 3 | PDF ingestion | done |
| 4 | Embeddings + policy FAISS index | done |
| 5 | Policy RAG | done |
| 6 | Company FAISS index | next |
| 7 | Portfolio database | pending |
| 8 | Portfolio-aware RAG | pending |
| 9 | News ingestion (RSS) | pending |
| 10 | Alert engine | pending |
| 11 | FastAPI endpoints | pending |
| 12 | React frontend | pending |
| 13 | Tests | pending |
| 14 | README + demo | pending |

Test suite currently: **139 passing** in the default run (about 25 s), plus 3
`slow` tests that run real local generation and are excluded unless you ask for
them with `pytest -m slow` (about 2 minutes). Live tests skip themselves when
Ollama is not running.

## 6. Current file structure

```
policy-investment-advisor/
├── .env, .env.example, .gitignore
├── Makefile, pytest.ini, requirements.txt, README.md
├── backend/
│   ├── __init__.py
│   ├── config.py              # typed settings from .env (singleton)
│   ├── logging_config.py      # idempotent logging setup
│   ├── exceptions.py          # error hierarchy with remediation strings
│   ├── main.py                # FastAPI factory, lifespan, error handler
│   ├── api/
│   │   ├── __init__.py
│   │   └── system.py          # /health, /api/config, /api/system/ollama
│   ├── rag/
│   │   ├── __init__.py
│   │   └── ollama_client.py   # health, model discovery, ChatOllama factory
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── metadata.py        # DocumentMetadata schema + date normalisation
│   │   ├── cleaning.py        # PDF text cleaning
│   │   ├── pdf_loader.py      # PDF -> one Document per page
│   │   └── chunker.py         # chunking + stable chunk ids + dedupe
│   ├── alerts/__init__.py     # Phase 10
│   ├── db/__init__.py         # Phase 7
│   └── vectorstore/
│       ├── policy_index/      # Phase 4
│       └── company_index/     # Phase 6
├── data/
│   ├── policies/              # source PDFs (RBI/SEBI/Budget)
│   ├── companies/             # reports, fundamentals
│   ├── news/                  # cached RSS articles
│   └── processed/
├── frontend/                  # directories reserved; built in Phase 12
│   └── src/{components,pages,services}/
├── scripts/
│   ├── check_setup.py         # Phase 1 verifier
│   ├── check_ollama.py        # Phase 2 verifier
│   ├── make_sample_pdf.py     # synthetic sample PDF + PDF writer
│   └── ingest_pdf.py          # Phase 3 ingestion preview
└── tests/
    ├── conftest.py            # pdf_factory fixture
    ├── test_phase1_scaffold.py
    ├── test_phase2_ollama.py
    └── test_phase3_ingestion.py
```

## 7. What each completed phase does

### Phase 1 — structure and configuration

`backend/config.py` exposes a Pydantic `Settings` class read from `.env` and
cached with `@lru_cache` as `get_settings()`. Nothing anywhere else hard-codes a
model name, URL or path. Relative paths resolve against the project root, and
`resolved_database_url` rewrites `sqlite:///data/app.db` to an absolute path so
the database does not move depending on the working directory.

`backend/logging_config.py` sets up root logging idempotently and quietens noisy
third-party loggers. The project uses `logging`, never `print`.

`scripts/check_setup.py` verifies Python version, directories, `.env`, imports
and config loading, treating the heavy RAG packages as a warning rather than a
failure.

### Phase 2 — local inference

`backend/rag/ollama_client.py` splits two concerns:

- **Inspection** via plain `requests` against Ollama's `/api/tags` — answers "is
  the server up?" and "which models are pulled?" without involving LangChain, so
  the health check survives a LangChain import problem.
- **Construction** via `build_chat_model()`, which assembles a `ChatOllama` from
  settings (model, base URL, temperature, `num_ctx`, timeout) without touching
  the network.

`check_status()` returns an `OllamaStatus` snapshot and never raises, so every
problem is reported at once rather than failing on the first.

Model name matching follows Ollama's own rule: a bare `qwen3` matches
`qwen3:latest` but deliberately not `qwen3:4b`, so `.env` and reality cannot
silently diverge.

`backend/exceptions.py` defines `AdvisorError` carrying a message, a
**remediation** string and an HTTP status. `ModelNotInstalledError("qwen3:4b")`
prints `Fix: Run: ollama pull qwen3:4b`. A global FastAPI exception handler turns
any of these into JSON with the fix included.

`/api/system/ollama` returns **200** when ready, **503** when the server is down,
**424** when a model needs pulling — so a caller can distinguish "start Ollama"
from "pull a model" without parsing strings. The API starts even when Ollama is
down; startup logs a warning instead of crashing.

### Phase 3 — PDF ingestion

`ingestion/metadata.py` — a `DocumentMetadata` dataclass rather than loose dicts.
`source` is mandatory, because a chunk nobody can cite is useless in a system
built on grounded answers. Fields: source, document_type, title, date, company,
sector, url, file_path, extra. Dates normalise to ISO from the formats Indian
regulators publish in (`10-08-2026`, `10 Aug 2026`).

`ingestion/cleaning.py` — PDF extraction produces text that is technically
correct and semantically poor: hyphenated words split across lines, running
headers on every page, newlines mid-sentence. A chunk containing
`"repo rate re-\nmains"` embeds further from a query about the repo rate than it
should. So: hyphen rejoining, soft-wrap unwrapping that preserves paragraph
breaks, page-number stripping, and `strip_repeated_lines`, which drops any line
appearing on more than 60% of pages as boilerplate (skipped under four pages,
where repetition is more likely genuine).

`ingestion/pdf_loader.py` — produces one `Document` per **page**, not per file,
because citations require page numbers and page identity is lost on
concatenation. Failed pages become empty strings rather than disappearing, so
list position always equals real page number. Encrypted PDFs get an
empty-password retry first, since many government PDFs are "protected" that way.

`ingestion/chunker.py` — `RecursiveCharacterTextSplitter` with separators ordered
so clause breaks are tried before sentence breaks; regulatory text is numbered
(`3.1`, `3.2`) and splitting mid-clause severs a rule from its number. Chunk ids
are content-hashed together with file path and page, so re-ingesting an unchanged
document yields identical ids — Phase 4 uses this to skip duplicates instead of
growing the index on every run.

Error handling: `InvalidDocumentError`, `EncryptedDocumentError`,
`EmptyDocumentError`. A scanned PDF is told to run `ocrmypdf`, not merely "no
text found".

`scripts/make_sample_pdf.py` writes a synthetic policy notice labelled
`SYNTHETIC SAMPLE DOCUMENT` on every page, and doubles as the test suite's PDF
writer so the project needs no PDF-generation dependency. The labelling is
deliberate: this system answers questions about real financial regulation, and a
realistic-looking fake circular in the index would eventually be retrieved and
cited as genuine.

## 8. Configuration (`.env`)

```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:4b
EMBEDDING_MODEL=embeddinggemma
LLM_TEMPERATURE=0.1
LLM_TIMEOUT_SECONDS=180
LLM_NUM_CTX=8192

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
MIN_RELEVANCE_SCORE=0.25

DATA_DIR=data
VECTORSTORE_DIR=backend/vectorstore
DATABASE_URL=sqlite:///data/app.db

API_HOST=127.0.0.1
API_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
LOG_LEVEL=INFO
```

## 9. How to run what exists

```bash
cd ~/Downloads/policy-investment-advisor
source .venv/bin/activate

python scripts/check_setup.py     # environment
python scripts/check_ollama.py    # local inference (Ollama must be running)
pytest                            # 66 tests

python scripts/make_sample_pdf.py
python scripts/ingest_pdf.py data/policies/SAMPLE_synthetic_policy_notice.pdf

uvicorn backend.main:app --reload  # then open http://127.0.0.1:8000/docs
```

## 10. Code quality rules in force

Type hints throughout; docstrings on anything non-obvious; `logging` not `print`;
LangChain-specific logic confined to `backend/rag/`; ingestion separate from
retrieval; database models will stay separate from API routes; FastAPI
dependency injection (`Depends(get_settings)`) rather than module globals; no
giant files; configuration never hard-coded.

One convention worth keeping: modules that need to be substitutable in tests are
imported **as modules** (`from backend.rag import ollama_client`) and called as
`ollama_client.check_status(...)`, because `from x import y` binds at import time
and cannot be monkeypatched.

## 11. Known issues and open decisions

- **Develop on `qwen3:1.7b`, judge quality on `qwen3:4b`.** Measured over the
  same corpus, retrieval and prompt: 1.7b averages 25.9 s against 4b's 124.6 s
  (max 69 s against 285 s). Answer-versus-refuse routing is identical — 1.7b
  refused every out-of-corpus question put to it — so the retrieval path, the
  relevance floors and the refusal logic can all be developed on it. What
  degrades is **grounding**: 1.7b cites loosely (it shotgun-cited all five
  sources on one question) and sometimes cites nothing at all, so
  **`grounded=False` is not trustworthy on 1.7b** — it is often a small-model
  artefact rather than a real ungrounded answer. `.env` ships with 1.7b; switch
  `LLM_MODEL` to `qwen3:4b` before judging answer quality or recording a demo.
- **Never tune `prompts.py` on 1.7b.** Any change to the grounding prompt, and
  any judgement about refusal or citation behaviour, must be verified on 4b.
  The two models fail differently, and a prompt tuned to satisfy the small one
  can quietly cost precision on the large one. Measure, do not assume: prompt
  wording dominates local latency here — an early five-rule draft of the system
  prompt generated 2237 tokens where the four-line version generates 259, which
  was 265 s against 45 s for the same question with the same context.
- **`LLM_TIMEOUT_SECONDS` must be raised to ~300 when running 4b.** At the
  default 180 s, 4b genuinely fails a question it can answer: "what is the
  penalty for premature withdrawal of a term deposit?" needs 285 s. 180 s is
  ample for 1.7b.
- **A streamed generation has no total time limit of its own.** The timeout in
  `client_kwargs` is an httpx timeout, and httpx bounds a *single read* — the
  gap between two tokens — not the call. `ChatOllama.invoke` streams
  internally, tokens arrive every few tens of milliseconds, and a 227 s
  generation therefore sailed straight past a 180 s setting. `invoke_messages`
  now consumes the stream itself and enforces a wall clock between chunks,
  raising `LLMTimeoutError` and closing the stream, which disconnects the client
  and is what actually tells Ollama to stop generating. The httpx timeout still
  covers the other case underneath: a server that never sends a first token.
- **Disabling Qwen3's thinking mode is counterproductive.** `reasoning=False`
  does not stop the model reasoning; it stops Ollama separating the reasoning
  out, so 1300-1900 characters of chain-of-thought land in the answer body.
  Leave `LLM_REASONING` unset and control length through the prompt.
- **`strip_repeated_lines` is heuristic.** Threshold 60% of pages, minimum four
  pages. If genuine content ever disappears, that is the knob.
- **Real circulars are bilingual.** RBI PDFs carry a Devanagari letterhead on
  every page. `strip_repeated_lines` removes it on documents of four pages or
  more (verified: 1 of 62 chunks on a 25-page Direction still carries any), but
  one- and two-page circulars fall under its minimum and keep theirs.
- **No absolute relevance score can separate the adjacent case.** Asking about
  NBFC deposit rules retrieves commercial-bank deposit rules at 0.539 — above
  three questions the corpus genuinely answers. `MIN_RELEVANCE_SCORE` (0.35)
  only rejects questions the corpus is not about at all; the prompt's
  applicability rule is what refuses the near-miss.
- **macOS 14 is past Homebrew's support window.** Ollama had to be installed from
  ollama.com rather than Homebrew, which wanted to compile it from source.
- **Table extraction loses column headers, so financial figures cannot be
  attributed to a year. No fix yet.** Extraction flattens a table into a run of
  numbers with the headers stripped, so a chunk retrieved from the ratios page
  of the Godrej annual report reads `Net profit margin (%) 7.83% 20.10` — two
  columns, FY24 and FY23, with nothing saying which is which. Neither the model
  nor a reader checking the citation can tell, so it will confidently state the
  wrong year's figure. **Financial questions against the company index are not
  trustworthy until this is fixed.** Likely direction: a table-aware extractor
  (`pdfplumber`, `camelot`) for pages detected as tabular, emitting one chunk
  per row with its headers attached, rather than letting `pypdf` flatten them.
- **Marketing prose outranks substantive disclosure in the company index. No
  fix yet.** Asking "what drove revenue growth?" returns "committed to driving
  category development through breakthrough innovation, robust brand building"
  at rank 1 — aspirational filler that embeds well against the question and
  says nothing — while the genuinely useful chunks (15% e-commerce growth in
  Africa and the USA, 41% of revenue international) rank 2 and 3. Annual
  reports are perhaps 40% this register, and nothing currently tells it apart
  from disclosure. Nothing in the retrieval policy helps: the floors filter by
  score, and the filler scores *well*. Possible directions, none tried: skip
  the front-of-book sections at ingest, weight chunks containing figures, or
  rerank retrieved chunks before they reach the model.
- **Company chunks score lower than policy chunks.** 0.38-0.47 against
  0.47-0.68, so the 0.35 absolute floor sits close to genuine matches and the
  relative ratio is doing nearly all the filtering. Worth rechecking when more
  companies are indexed.
- **pypdf warns about CFF Type1 font encoding** on the Godrej report
  (`fontTools is required to fully parse...`). Text extracted correctly, so
  `fonttools` is not installed; if mangled characters appear later, that is the
  first thing to add.

## 12. What Phase 4 must deliver

1. An embeddings wrapper over `OllamaEmbeddings` reading `EMBEDDING_MODEL` from
   config, with batching and clear errors when the model is missing.
2. A FAISS index builder that persists to `backend/vectorstore/policy_index/`
   and reloads from disk.
3. Deduplication on re-ingest using the existing `chunk_id`.
4. Similarity search returning documents with scores and full metadata intact.
5. A CLI to build the policy index from `data/policies/` and query it.
6. Tests covering embedding dimensions, index round-trip through disk, metadata
   preservation and retrieval ordering.
