"""Phase 9 tool: fetch RSS news and build the news index.

Reads the feeds declared in ``data/news/feeds.json``, attributes each article to
a company from the Phase 6 registry, and indexes it in
``backend/vectorstore/news_index/`` — separate from the company index, so a
589-page filing cannot crowd out a 300-word story.

Only what a feed publishes is read; article pages are never fetched. A feed that
is down, malformed or empty is reported and skipped, never fatal.

Run with::

    python scripts/build_news_index.py feeds        # what is declared, and does it work
    python scripts/build_news_index.py build
    python scripts/build_news_index.py build --limit 10 --dry-run
    python scripts/build_news_index.py query "repo rate decision"
    python scripts/build_news_index.py query "input costs" --company GODREJCP
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
from backend.ingestion.company_registry import CompanyRegistry  # noqa: E402
from backend.ingestion.news import attribution_counts, fetch_all, load_feeds  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.rag import embeddings as embeddings_module  # noqa: E402
from backend.rag.retrieval import NewsRetriever, freshness  # noqa: E402
from backend.rag.vector_store import news_index  # noqa: E402

PASS, FAIL, INFO, WARN = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]"
PREVIEW_CHARS = 180


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def feeds(args: argparse.Namespace, settings: Settings) -> int:
    """List the declared feeds and check each one responds."""
    declared = load_feeds(settings=settings)
    if not declared:
        _report(FAIL, f"No feeds declared in {settings.news_data_dir / 'feeds.json'}")
        return 1

    registry = CompanyRegistry.load(settings=settings)
    _report(INFO, f"{len(declared)} feed(s), {len(registry)} companies for attribution")

    _, results = fetch_all(declared, registry, limit=args.limit)
    for result in results:
        if result.ok:
            _report(PASS, f"{result.feed.name}: {len(result.articles)} articles "
                          f"({result.skipped} skipped) in {result.elapsed_seconds:.1f}s")
        else:
            _report(WARN, f"{result.feed.name}: {result.error}")
    return 0


def build(args: argparse.Namespace, settings: Settings) -> int:
    """Fetch every feed and add what is new to the index."""
    declared = load_feeds(settings=settings)
    if not declared:
        _report(FAIL, f"No feeds declared in {settings.news_data_dir / 'feeds.json'}")
        return 1

    registry = CompanyRegistry.load(settings=settings)
    index = news_index(settings=settings)

    _report(INFO, f"Feeds: {len(declared)} | index: {index.directory}")

    if args.rebuild and index.exists:
        index.clear()
        _report(INFO, "Cleared the existing index (--rebuild)")

    limit = args.limit or settings.news_max_articles_per_feed
    articles, results = fetch_all(declared, registry, limit=limit)

    for result in results:
        if not result.ok:
            _report(WARN, f"{result.feed.name} unavailable: {result.error}")

    working = sum(1 for result in results if result.ok)
    if not articles:
        _report(FAIL, f"No articles gathered ({working}/{len(results)} feeds responded)")
        return 1

    _report(PASS, f"{len(articles)} articles from {working}/{len(results)} feeds")
    _report(INFO, f"  attribution: {attribution_counts(articles)}")
    dated = sum(1 for a in articles if a.metadata.get("date"))
    _report(INFO, f"  dated: {dated}/{len(articles)} (undated articles do not decay)")

    if args.dry_run:
        for article in articles[:5]:
            meta = article.metadata
            sys.stdout.write(
                f"\n  {meta.get('date') or 'no date':10} {meta.get('source')} "
                f"[{meta.get('company') or '-'}]\n"
                f"    {' '.join(article.page_content.split())[:PREVIEW_CHARS]}...\n"
            )
        _report(INFO, "Dry run: nothing indexed")
        return 0

    try:
        embeddings_module.verify_ready(settings)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    chunks = deduplicate(chunk_documents(articles, settings))
    _report(PASS, f"Prepared {len(chunks)} chunks")

    try:
        report = index.add_documents(chunks)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    if report.added:
        _report(PASS, f"Added {report.added} chunks in {report.elapsed_seconds:.1f}s")
    if report.skipped:
        _report(PASS, f"Skipped {report.skipped} already indexed (same article, same run or earlier)")

    stats = index.stats()
    _report(PASS, f"Index now holds {stats['vector_count']} vectors")
    return 0


def query(args: argparse.Namespace, settings: Settings) -> int:
    """Search the news index, ranked by relevance and recency."""
    retriever = NewsRetriever(settings=settings)
    _report(INFO, f'Query: "{args.text}"')
    _report(
        INFO,
        f"half-life {settings.news_half_life_days}d, floor {settings.news_freshness_floor} "
        "(news only; policy and filings never decay)",
    )

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

    _report(PASS, f"{len(retrieval.results)} result(s) in {elapsed:.2f}s")
    for rank, result in enumerate(retrieval.results, start=1):
        meta = result.metadata
        weight = freshness(
            meta,
            half_life_days=settings.news_half_life_days,
            floor=settings.news_freshness_floor,
        )
        sys.stdout.write("\n" + "-" * 74 + "\n")
        sys.stdout.write(
            f"{rank}. raw {result.score:.3f} x freshness {weight:.2f} = {result.score * weight:.3f}"
            f" | {meta.get('date') or 'no date'} | {meta.get('company') or '-'}\n"
        )
        sys.stdout.write(f"   {meta.get('source')}: {meta.get('title', '')[:90]}\n")
        if meta.get("url"):
            sys.stdout.write(f"   {meta['url']}\n")
    sys.stdout.write("\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch RSS news and build the news index.")
    sub = parser.add_subparsers(dest="command", required=True)

    lister = sub.add_parser("feeds", help="List declared feeds and check they respond")
    lister.add_argument("--limit", type=int, default=5, help="Entries to read per feed")
    lister.set_defaults(handler=feeds)

    builder = sub.add_parser("build", help="Fetch feeds and index new articles")
    builder.add_argument("--limit", type=int, default=None, help="Articles per feed")
    builder.add_argument("--rebuild", action="store_true", help="Delete the index first")
    builder.add_argument("--dry-run", action="store_true", help="Fetch and report, index nothing")
    builder.set_defaults(handler=build)

    searcher = sub.add_parser("query", help="Search the news index")
    searcher.add_argument("text", help="Natural-language question")
    searcher.add_argument("--company", default=None, help="Restrict to this ticker")
    searcher.add_argument("--sector", default=None, help="Restrict to this sector")
    searcher.add_argument("--k", type=int, default=None, help="Results to return")
    searcher.set_defaults(handler=query)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    sys.stdout.write(f"Phase 9 news index — {args.command}\n" + "-" * 74 + "\n")
    try:
        return int(args.handler(args, settings))
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
