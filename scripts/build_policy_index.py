"""Phase 4 tool: build and query the policy FAISS index.

Reads every PDF under ``data/policies/``, chunks it with the Phase 3 pipeline,
embeds the chunks with the local embedding model and stores them in
``backend/vectorstore/policy_index/``. Re-running skips chunks already indexed,
so adding one new circular costs one circular's worth of embedding, not the
whole corpus.

Run with::

    python scripts/build_policy_index.py build
    python scripts/build_policy_index.py build --rebuild          # from scratch
    python scripts/build_policy_index.py query "repo rate decision" --k 5
    python scripts/build_policy_index.py stats
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Settings, get_settings  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.ingestion.chunker import chunk_documents, deduplicate  # noqa: E402
from backend.ingestion.metadata import DocumentMetadata  # noqa: E402
from backend.ingestion.pdf_loader import iter_pdfs, load_pdf  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.rag import embeddings as embeddings_module  # noqa: E402
from backend.rag.vector_store import VectorIndex, policy_index  # noqa: E402

PASS, FAIL, INFO = "[ OK ]", "[FAIL]", "[INFO]"
PREVIEW_CHARS = 260

#: Publishers recognised from a filename prefix, e.g. ``RBI_master_circular.pdf``.
#: Anything else falls back to ``--source``, because a chunk without a real
#: publisher cannot be cited honestly.
KNOWN_PUBLISHERS = {"RBI", "SEBI", "IRDAI", "MOF", "PIB", "BUDGET", "SAMPLE"}


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def infer_source(path: Path, fallback: str) -> str:
    """Derive the publisher from a filename prefix, e.g. ``SEBI_circular.pdf``."""
    prefix = path.stem.split("_", 1)[0].upper()
    return prefix if prefix in KNOWN_PUBLISHERS else fallback


def build(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest every policy PDF into the index."""
    directory = Path(args.directory) if args.directory else settings.policy_data_dir
    index = policy_index(settings=settings)

    _report(INFO, f"Source directory: {directory}")
    _report(INFO, f"Index directory:  {index.directory}")
    _report(INFO, f"Embedding model:  {settings.embedding_model}")

    if args.rebuild and index.exists:
        index.clear()
        _report(INFO, "Cleared the existing index (--rebuild)")

    try:
        embeddings_module.verify_ready(settings)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    _report(PASS, "Embedding model is installed")

    pdfs = list(iter_pdfs(directory))
    if not pdfs:
        _report(FAIL, f"No PDFs found in {directory}")
        _report(INFO, "Add circulars there, or generate one: python scripts/make_sample_pdf.py")
        return 1
    _report(PASS, f"Found {len(pdfs)} PDF(s)")

    chunks = []
    for pdf in pdfs:
        source = infer_source(pdf, args.source)
        try:
            pages = load_pdf(pdf, DocumentMetadata(source=source, document_type=args.document_type))
        except AdvisorError as exc:
            _report(FAIL, f"Skipped {pdf.name}: {exc.message}")
            continue
        file_chunks = chunk_documents(pages, settings)
        chunks.extend(file_chunks)
        _report(INFO, f"  {pdf.name}: {len(pages)} pages -> {len(file_chunks)} chunks [{source}]")

    chunks = deduplicate(chunks)
    if not chunks:
        _report(FAIL, "Nothing to index — every PDF failed or was empty")
        return 1
    _report(PASS, f"Prepared {len(chunks)} unique chunks")

    _report(INFO, "Embedding (first call loads the model into RAM; this is the slow part)...")
    try:
        report = index.add_documents(chunks)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    if report.added:
        rate = report.added / report.elapsed_seconds if report.elapsed_seconds else 0.0
        _report(PASS, f"Added {report.added} chunks in {report.elapsed_seconds:.1f}s ({rate:.1f}/s)")
    if report.skipped:
        _report(PASS, f"Skipped {report.skipped} chunks already in the index")

    stats = index.stats()
    _report(PASS, f"Index now holds {stats['vector_count']} vectors of dimension {stats['dimension']}")
    _report(INFO, f"  sources: {stats['sources']}")
    return 0


def query(args: argparse.Namespace, settings: Settings) -> int:
    """Run one similarity search against the built index."""
    index = policy_index(settings=settings)
    min_score = settings.min_relevance_score if args.min_score is None else args.min_score

    _report(INFO, f'Query: "{args.text}"')
    _report(INFO, f"top_k={args.k}, min_score={min_score}")

    started = time.perf_counter()
    try:
        results = index.search(args.text, k=args.k, min_score=min_score)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    elapsed = time.perf_counter() - started

    if not results:
        _report(FAIL, f"No chunk scored above {min_score} (searched in {elapsed:.2f}s)")
        _report(INFO, "Lower the bar to inspect what is there: --min-score 0")
        return 1

    _report(PASS, f"{len(results)} result(s) in {elapsed:.2f}s")
    for rank, result in enumerate(results, start=1):
        sys.stdout.write("\n" + "-" * 70 + "\n")
        sys.stdout.write(f"{rank}. score {result.score:.3f} | {result.citation()}\n")
        sys.stdout.write(f"   chunk {result.metadata.get('chunk_id')}\n")
        preview = " ".join(result.text.split())[:PREVIEW_CHARS]
        sys.stdout.write(f"   {preview}...\n")
    sys.stdout.write("\n")
    return 0


def stats(args: argparse.Namespace, settings: Settings) -> int:
    """Print what the index currently holds."""
    index: VectorIndex = policy_index(settings=settings)
    try:
        summary = index.stats()
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    if not summary["built"]:
        _report(FAIL, f"No index at {summary['directory']}")
        _report(INFO, "Build it: python scripts/build_policy_index.py build")
        return 1

    _report(PASS, f"{summary['vector_count']} vectors of dimension {summary['dimension']}")
    _report(INFO, f"  directory: {summary['directory']}")
    _report(INFO, f"  documents: {summary['document_count']}")
    _report(INFO, f"  sources:   {summary['sources']}")
    manifest = summary.get("manifest") or {}
    _report(INFO, f"  model:     {manifest.get('embedding_model', 'unknown')}")
    _report(INFO, f"  updated:   {manifest.get('updated_at', 'unknown')}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query the policy FAISS index.")
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="Ingest data/policies/ into the index")
    builder.add_argument("--directory", default=None, help="Override the source directory")
    builder.add_argument("--rebuild", action="store_true", help="Delete the index first")
    builder.add_argument("--source", default="POLICY", help="Publisher when the filename implies none")
    builder.add_argument("--type", dest="document_type", default="policy", help="Document type")
    builder.set_defaults(handler=build)

    searcher = sub.add_parser("query", help="Search the index")
    searcher.add_argument("text", help="Natural-language question")
    searcher.add_argument("--k", type=int, default=None, help="Results to return")
    searcher.add_argument("--min-score", type=float, default=None, help="Cosine similarity floor")
    searcher.set_defaults(handler=query)

    reporter = sub.add_parser("stats", help="Describe the index on disk")
    reporter.set_defaults(handler=stats)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    if getattr(args, "k", None) is None and args.command == "query":
        args.k = settings.retrieval_top_k

    sys.stdout.write(f"Phase 4 policy index — {args.command}\n" + "-" * 70 + "\n")
    return int(args.handler(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
