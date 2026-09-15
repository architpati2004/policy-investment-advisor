# Policy-Based Investment Advisor (RAG)

A fully local, retrieval-augmented research assistant that contextualises government
and regulatory policy (RBI, SEBI, Union Budget, circulars) plus company and news
information against a user's investment portfolio.

Everything runs offline on an Apple Silicon MacBook Air using **Ollama** for
inference and **FAISS** for vector search. No OpenAI key, no Anthropic key, no paid
database, no paid news API.

> This is an educational research tool. Outputs are analysis grounded in retrieved
> sources, not guaranteed financial advice.

---

## Stack

| Layer | Choice |
| --- | --- |
| LLM | Ollama + Qwen (`qwen3:4b` by default) |
| Embeddings | Ollama (`embeddinggemma`, or `qwen3-embedding`) |
| Vector store | FAISS — two independent indexes (policy, company) |
| Orchestration | LangChain |
| Backend | FastAPI + Pydantic |
| Database | SQLite + SQLAlchemy |
| Frontend | React + Vite |

---

## Build phases

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Project structure + environment setup | ✅ done |
| 2 | Local Ollama connection | ✅ done |
| 3 | PDF ingestion | ✅ done |
| 4 | Embeddings + policy FAISS index | pending |
| 5 | Policy RAG | pending |
| 6 | Company FAISS index | pending |
| 7 | Portfolio database | pending |
| 8 | Portfolio-aware RAG | pending |
| 9 | News ingestion (RSS) | pending |
| 10 | Alert engine | pending |
| 11 | FastAPI endpoints | pending |
| 12 | React frontend | pending |
| 13 | Tests | pending |
| 14 | README + demo instructions | pending |

---

## Prerequisites

1. **Python 3.11 or 3.12**

   ```bash
   python3 --version
   ```

2. **Ollama** (installed now, exercised in Phase 2)

   ```bash
   brew install ollama
   ollama serve          # leave running in its own terminal
   ```

3. **Models** — pull these once (roughly 2.5 GB + 0.6 GB):

   ```bash
   ollama pull qwen3:4b
   ollama pull embeddinggemma
   ```

   On a low-RAM machine use `qwen3:1.7b` instead and set `LLM_MODEL` in `.env`.
   If `embeddinggemma` is unavailable in your Ollama build, use
   `qwen3-embedding:0.6b` and set `EMBEDDING_MODEL` accordingly. Model names are
   read from configuration and appear nowhere else in the code.

4. **Node 18+** (only needed from Phase 12 onwards).

---

## Setup

```bash
cd policy-investment-advisor

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Equivalent shortcuts: `make setup` and `make env`.

### Verify the scaffold

```bash
python scripts/check_setup.py
```

Expected output ends with `Phase 1 scaffold is ready.`

### Run the API

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl http://127.0.0.1:8000/api/config
# {"app_name":"...","llm_model":"qwen3:4b","embedding_model":"embeddinggemma",...}
```

Interactive docs: <http://127.0.0.1:8000/docs>

### Verify local inference (Phase 2)

With `ollama serve` running in another terminal:

```bash
python scripts/check_ollama.py
```

It reports the server status, which models are pulled, and how long a single
generation takes. Exit code 0 means local inference is ready.

The same information is available from the API at `/api/system/ollama`:
200 when ready, 503 when the server is down, 424 when a model needs pulling.

### Ingest a PDF (Phase 3)

Generate a synthetic sample document, then preview what ingestion does to it:

```bash
python scripts/make_sample_pdf.py
python scripts/ingest_pdf.py data/policies/SAMPLE_synthetic_policy_notice.pdf
```

Point it at a real circular once you have one:

```bash
python scripts/ingest_pdf.py data/policies/rbi_circular.pdf \
  --source RBI --type circular --date 2026-08-10
```

The sample is labelled SYNTHETIC on every page and should be deleted once real
documents are in place — a realistic-looking fake circular in the index would
eventually be retrieved and cited as genuine.

### Run the tests

```bash
pytest
```

---

## Configuration

All settings live in `.env` and are loaded by `backend/config.py` into a typed
`Settings` object. Nothing is hard-coded elsewhere.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `LLM_MODEL` | `qwen3:4b` | Generation model |
| `EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `LLM_TEMPERATURE` | `0.1` | Low, to reduce fabrication |
| `LLM_TIMEOUT_SECONDS` | `180` | Guards slow local inference |
| `LLM_NUM_CTX` | `8192` | Context window |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text splitting |
| `RETRIEVAL_TOP_K` | `5` | Chunks per retriever |
| `MIN_RELEVANCE_SCORE` | `0.25` | Floor below which a chunk is discarded |
| `DATA_DIR` | `data` | Source documents |
| `VECTORSTORE_DIR` | `backend/vectorstore` | FAISS indexes |
| `DATABASE_URL` | `sqlite:///data/app.db` | SQLite database |

Relative paths resolve against the project root, so the app behaves the same
regardless of the working directory you launch it from.

---

## Project layout

```
policy-investment-advisor/
├── backend/
│   ├── main.py              # FastAPI entrypoint (app factory + lifespan)
│   ├── config.py            # Typed settings loaded from .env
│   ├── logging_config.py    # Central logging setup
│   ├── api/                 # Routers          (Phase 11)
│   ├── rag/                 # LangChain logic  (Phases 4-8)
│   ├── ingestion/           # Loaders/chunking (Phases 3, 6, 9)
│   ├── alerts/              # Alert engine     (Phase 10)
│   ├── db/                  # SQLAlchemy       (Phase 7)
│   └── vectorstore/
│       ├── policy_index/    # FAISS: RBI / SEBI / Budget / circulars
│       └── company_index/   # FAISS: fundamentals / reports / news
├── data/
│   ├── policies/  companies/  news/  processed/
├── frontend/
│   └── src/{components,pages,services}/   # Vite app scaffolded in Phase 12
├── scripts/check_setup.py
├── tests/
├── requirements.txt
├── .env.example
└── Makefile
```

### Architectural boundaries

- LangChain imports are confined to `backend/rag/`.
- Ingestion never imports retrieval, and vice versa.
- SQLAlchemy models stay out of API route modules.
- Frontend HTTP calls stay in `frontend/src/services/`.

---

## Make targets

```
make setup     # venv + dependencies
make env       # create .env from the example
make models    # ollama pull qwen3:4b embeddinggemma
make run       # start the dev server
make test      # run pytest
make verify    # run the setup checker
make clean     # remove venv and caches
```
