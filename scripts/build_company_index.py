"""Phase 6 tool: build and query the company FAISS index.

Reads company filings from ``data/companies/``, attributes each to a company
declared in ``data/companies/registry.json``, and stores the chunks in
``backend/vectorstore/company_index/``. As with the policy index, re-running
only embeds what is new.

Queries can be scoped to holdings, which is what Phase 8 will do with a real
portfolio::

    python scripts/build_company_index.py build
    python scripts/build_company_index.py build --company GODREJCP   # force attribution
    python scripts/build_company_index.py query "what drove revenue growth?"
    python scripts/build_company_index.py query "margin pressure" --company GODREJCP
    python scripts/build_company_index.py query "input costs" --sector FMCG
    python scripts/build_company_index.py companies
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
from backend.ingestion.company_docs import companies_in, load_company_directory  # noqa: E402
from backend.ingestion.company_registry import Company, CompanyRegistry  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.rag import embeddings as embeddings_module  # noqa: E402
from backend.rag.retrieval import CompanyRetriever  # noqa: E402
from backend.rag.vector_store import company_index  # noqa: E402

PASS, FAIL, INFO, WARN = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]"
PREVIEW_CHARS = 220


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def _override(args: argparse.Namespace, registry: CompanyRegistry) -> Company | None:
    """Turn ``--company``/``--sector`` into a company, if given.

    A ticker already in the registry is looked up so its sector and aliases are
    not lost; an unknown ticker needs ``--sector`` too, since a company chunk
    without a sector cannot be matched to a holding by sector later.
    """
    if not args.company:
        return None
    existing = registry.by_ticker(args.company)
    if existing and not args.sector:
        return existing
    sector = args.sector or (existing.sector if existing else None)
    if not sector:
        raise AdvisorError(
            f"'{args.company}' is not in the registry, so --sector is required",
            remediation="Add it to data/companies/registry.json, or pass --sector.",
        )
    return Company(
        ticker=args.company,
        name=existing.name if existing else args.company,
        sector=sector,
    )


def build(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest company filings into the index."""
    directory = Path(args.directory) if args.directory else settings.company_data_dir
    registry = CompanyRegistry.load(settings=settings)
    index = company_index(settings=settings)

    _report(INFO, f"Source directory: {directory}")
    _report(INFO, f"Index directory:  {index.directory}")
    _report(INFO, f"Registry:         {len(registry)} companies, sectors {registry.sectors()}")

    if args.rebuild and index.exists:
        index.clear()
        _report(INFO, "Cleared the existing index (--rebuild)")

    try:
        embeddings_module.verify_ready(settings)
        override = _override(args, registry)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    _report(PASS, "Embedding model is installed")

    pages, skipped = load_company_directory(
        directory, registry, document_type=args.document_type, override=override
    )
    for miss in skipped:
        _report(WARN, f"Skipped {miss.path.name}: {miss.reason}")

    if not pages:
        _report(FAIL, f"Nothing to index from {directory}")
        _report(INFO, "Add filings there and declare them in data/companies/registry.json")
        return 1
    _report(PASS, f"Loaded {len(pages)} pages; per company: {companies_in(pages)}")

    chunks = deduplicate(chunk_documents(pages, settings))
    _report(PASS, f"Prepared {len(chunks)} unique chunks")
    _report(INFO, "Embedding (this is the slow part; a full annual report takes minutes)...")

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
    return 0


def query(args: argparse.Namespace, settings: Settings) -> int:
    """Search the company index, optionally scoped to holdings."""
    retriever = CompanyRetriever(settings=settings)
    scope = []
    if args.company:
        scope.append(f"company {args.company}")
    if args.sector:
        scope.append(f"sector {args.sector}")
    _report(INFO, f'Query: "{args.text}"' + (f" | scope: {', '.join(scope)}" if scope else ""))

    started = time.perf_counter()
    try:
        retrieval = retriever.retrieve(
            args.text,
            companies=[args.company] if args.company else None,
            sectors=[args.sector] if args.sector else None,
            k=args.k,
        )
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    elapsed = time.perf_counter() - started

    if not retrieval.has_context:
        _report(FAIL, f"Nothing relevant in {elapsed:.2f}s — {retrieval.reason}")
        return 0

    _report(PASS, f"{len(retrieval.results)} result(s) in {elapsed:.2f}s "
                  f"(considered {retrieval.considered}, floor {retrieval.floor:.3f})")
    for rank, result in enumerate(retrieval.results, start=1):
        meta = result.metadata
        sys.stdout.write("\n" + "-" * 70 + "\n")
        sys.stdout.write(
            f"{rank}. score {result.score:.3f} | {meta.get('company')} / {meta.get('sector')}"
            f" | {result.citation()}\n"
        )
        preview = " ".join(result.text.split())[:PREVIEW_CHARS]
        sys.stdout.write(f"   {preview}...\n")
    sys.stdout.write("\n")
    return 0


def companies(args: argparse.Namespace, settings: Settings) -> int:
    """Show the declared companies and what the index actually holds for each."""
    registry = CompanyRegistry.load(settings=settings)
    index = company_index(settings=settings)

    _report(INFO, f"Registry: {len(registry)} companies")
    for company in registry:
        aliases = f" (aliases: {', '.join(company.aliases)})" if company.aliases else ""
        sys.stdout.write(f"  {company.ticker:12} {company.sector:14} {company.name}{aliases}\n")

    if not index.exists:
        _report(WARN, "The company index has not been built yet")
        return 0

    try:
        index.load()
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    indexed = companies_in(index.documents())
    _report(PASS, f"Indexed chunks per company: {indexed}")
    declared = {company.ticker for company in registry}
    for ticker in indexed:
        if ticker not in declared:
            _report(WARN, f"'{ticker}' is in the index but not in the registry")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query the company FAISS index.")
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="Ingest data/companies/ into the index")
    builder.add_argument("--directory", default=None, help="Override the source directory")
    builder.add_argument("--rebuild", action="store_true", help="Delete the index first")
    builder.add_argument("--company", default=None, help="Attribute every file to this ticker")
    builder.add_argument("--sector", default=None, help="Sector, when --company is not in the registry")
    builder.add_argument("--type", dest="document_type", default="annual_report", help="Filing type")
    builder.set_defaults(handler=build)

    searcher = sub.add_parser("query", help="Search the index")
    searcher.add_argument("text", help="Natural-language question")
    searcher.add_argument("--company", default=None, help="Restrict to this ticker")
    searcher.add_argument("--sector", default=None, help="Restrict to this sector")
    searcher.add_argument("--k", type=int, default=None, help="Chunks to retrieve")
    searcher.set_defaults(handler=query)

    lister = sub.add_parser("companies", help="List declared and indexed companies")
    lister.set_defaults(handler=companies)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    sys.stdout.write(f"Phase 6 company index — {args.command}\n" + "-" * 70 + "\n")
    return int(args.handler(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
