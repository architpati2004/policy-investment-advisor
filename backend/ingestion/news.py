"""News ingestion from RSS feeds.

Feeds are the one part of this system that is guaranteed to be broken at any
given moment: publishers change URLs, ship invalid XML, drop fields, return HTML
error pages with a 200, and go down. So every failure here is per-feed and
reported, never fatal — one dead feed must not cost the run the other nine.

**Only what the feed itself publishes is read.** An RSS feed is offered for
syndication; the article page behind it usually is not, and fetching it would
mean ignoring whatever the site's terms and robots rules say. What that costs is
visible in the feeds themselves: RBI puts a full 5,000-character press release in
its ``summary``, Business Standard puts 178 characters, NDTV Profit 102. So some
articles arrive as substance and others as a headline and a link. A headline is
still worth indexing — it dates a development and points at the source — but it
will not answer a question on its own, and it is not padded out to pretend
otherwise.

``bozo`` is not treated as failure. RBI's own feed sets it and still yields ten
perfectly good entries; malformed-but-parseable is the normal case, so the test
that matters is whether entries came back, not whether the XML validated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import feedparser
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from backend.config import Settings, get_settings
from backend.exceptions import InvalidFeedsError
from backend.ingestion.cleaning import clean_text
from backend.ingestion.company_registry import Company, CompanyRegistry
from backend.ingestion.metadata import DocumentMetadata
from backend.logging_config import get_logger

logger = get_logger(__name__)

FEEDS_FILENAME = "feeds.json"
#: Identifies this client to publishers rather than pretending to be a browser.
USER_AGENT = "policy-investment-advisor/0.1 (local research tool)"
#: Below this there is nothing to embed — a stub like "Markets" with no body.
#: Deliberately low: a headline-only entry still dates a development and points
#: at its source, and feeds that ship 102-character summaries are common. A
#: higher bar would silently drop real articles for being tersely written.
MIN_ARTICLE_CHARS = 20


@dataclass(frozen=True)
class Feed:
    """One RSS feed, as declared in ``data/news/feeds.json``."""

    name: str
    url: str
    #: Optional: tags every article from this feed, for feeds that are
    #: sector-specific ("banking news"). Company attribution still comes from
    #: the registry.
    sector: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Feed:
        if not isinstance(payload, dict):
            raise InvalidFeedsError(f"expected an object per feed, got {type(payload).__name__}")
        name = str(payload.get("name", "")).strip()
        url = str(payload.get("url", "")).strip()
        if not name or not url:
            raise InvalidFeedsError(f"each feed needs 'name' and 'url' (got {payload!r})")
        if not url.startswith(("http://", "https://")):
            raise InvalidFeedsError(f"feed url must be http(s): {url!r}")
        sector = str(payload.get("sector", "")).strip() or None
        return cls(name=name, url=url, sector=sector)


@dataclass(frozen=True)
class FeedResult:
    """What one feed produced, including why it produced nothing."""

    feed: Feed
    articles: list[Document] = field(default_factory=list)
    error: str | None = None
    skipped: int = 0
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def load_feeds(path: str | Path | None = None, settings: Settings | None = None) -> list[Feed]:
    """Read the feed list. A missing file yields an empty list, not an error."""
    settings = settings or get_settings()
    path = Path(path) if path is not None else settings.news_data_dir / FEEDS_FILENAME

    if not path.is_file():
        logger.info("No feed list at %s", path)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise InvalidFeedsError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise InvalidFeedsError("the feed list must be a JSON list of feed objects")

    feeds = [Feed.from_dict(entry) for entry in data]
    logger.info("Loaded %d feeds from %s", len(feeds), path)
    return feeds


def clean_article(raw: str) -> str:
    """Strip feed HTML down to readable text.

    Summaries arrive as HTML fragments — links, entities, the occasional tracking
    pixel. The tags are removed first, then the same cleaning the PDF pipeline
    uses runs over the result, so an article and a circular reach the embedding
    model looking alike.
    """
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    return clean_text(text)


def parse_published(entry: Any) -> str | None:
    """Publication date as ``YYYY-MM-DD``, or ``None`` when the feed omits it.

    Tries feedparser's own parse first, then falls back to reading the raw date
    string, because feedparser is stricter than publishers are. RBI stamps its
    press releases ``Wed, 16 Sep 2026 14:30:00`` with no timezone, which RFC 822
    requires; feedparser therefore returns ``None`` and every RBI article would
    arrive undated, which is precisely the metadata recency ranking runs on.

    ``None`` is left as ``None`` rather than defaulted to today. Recency scoring
    treats an undated article as undecayed, and stamping it with the fetch date
    would be inventing a fact — the citation says "no date" instead.
    """
    for key in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, key, None) or (entry.get(key) if hasattr(entry, "get") else None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc).date().isoformat()
            except (TypeError, ValueError):
                continue

    for key in ("published", "updated"):
        raw = entry.get(key) if hasattr(entry, "get") else getattr(entry, key, None)
        if not raw:
            continue
        try:
            return parsedate_to_datetime(str(raw)).date().isoformat()
        except (TypeError, ValueError):
            continue
    return None


def article_document(
    entry: Any,
    feed: Feed,
    registry: CompanyRegistry,
) -> Document | None:
    """Turn one feed entry into a Document, or ``None`` if there is nothing to index.

    Company attribution reuses the Phase 6 registry — the same declarations the
    company index and the portfolio already key on — rather than a second mapping
    that could disagree with it. An article naming several holdings is attributed
    to the most specific match and records the rest in ``mentions``.
    """
    title = clean_article(str(entry.get("title", ""))).strip()
    body = clean_article(str(entry.get("summary", "") or entry.get("description", "")))
    url = str(entry.get("link", "")).strip() or None

    text = f"{title}. {body}".strip(". ").strip() if title else body
    if len(text) < MIN_ARTICLE_CHARS:
        return None

    mentioned = registry.mentioned_in(f"{title} {body}")
    primary: Company | None = mentioned[0] if mentioned else None

    extra: dict[str, Any] = {}
    if len(mentioned) > 1:
        # Flat metadata, so a list is stored as a string. Kept because an
        # article about two holdings is attributed to one of them, and the other
        # would otherwise vanish.
        extra["mentions"] = ",".join(company.ticker for company in mentioned)

    metadata = DocumentMetadata(
        source=feed.name,
        document_type="news",
        title=title or (url or "untitled"),
        date=parse_published(entry),
        company=primary.ticker if primary else None,
        sector=primary.sector if primary else feed.sector,
        url=url,
        extra=extra,
    )
    return Document(page_content=text, metadata=metadata.to_dict())


def fetch_feed(
    feed: Feed,
    registry: CompanyRegistry,
    *,
    limit: int | None = None,
    parser: Any = None,
) -> FeedResult:
    """Fetch and parse one feed. Never raises.

    Args:
        feed: The feed to read.
        registry: Declared companies, for attribution.
        limit: Keep at most this many entries.
        parser: Override for ``feedparser.parse``, so tests need no network.

    Returns:
        A result carrying either articles or the reason there are none.
    """
    started = time.perf_counter()
    parse = parser or feedparser.parse

    try:
        parsed = parse(feed.url, agent=USER_AGENT)
    except Exception as exc:  # noqa: BLE001 - a broken feed must not end the run
        logger.error("Feed %s failed: %s", feed.name, exc)
        return FeedResult(feed, error=str(exc), elapsed_seconds=time.perf_counter() - started)

    status = getattr(parsed, "status", None)
    entries = list(getattr(parsed, "entries", []) or [])

    if status is not None and int(status) >= 400:
        return FeedResult(
            feed,
            error=f"HTTP {status}",
            elapsed_seconds=time.perf_counter() - started,
        )

    if not entries:
        # Only now does bozo matter: it explains an empty result rather than
        # condemning a malformed feed that parsed anyway.
        detail = getattr(parsed, "bozo_exception", None)
        reason = f"no entries ({detail})" if detail else "no entries"
        return FeedResult(feed, error=reason, elapsed_seconds=time.perf_counter() - started)

    articles: list[Document] = []
    skipped = 0
    for entry in entries[: limit or len(entries)]:
        try:
            document = article_document(entry, feed, registry)
        except Exception as exc:  # noqa: BLE001 - one bad entry, not the whole feed
            logger.warning("Entry in %s could not be read: %s", feed.name, exc)
            skipped += 1
            continue
        if document is None:
            skipped += 1
            continue
        articles.append(document)

    elapsed = time.perf_counter() - started
    logger.info(
        "Feed %s: %d articles, %d skipped in %.1fs", feed.name, len(articles), skipped, elapsed
    )
    return FeedResult(feed, articles=articles, skipped=skipped, elapsed_seconds=elapsed)


def fetch_all(
    feeds: Iterable[Feed],
    registry: CompanyRegistry,
    *,
    limit: int | None = None,
    parser: Any = None,
) -> tuple[list[Document], list[FeedResult]]:
    """Fetch every feed, collecting failures rather than stopping on them.

    Returns:
        ``(articles, results)`` — everything gathered, and a per-feed report so
        the caller can show which feeds are broken.
    """
    articles: list[Document] = []
    results: list[FeedResult] = []

    for feed in feeds:
        result = fetch_feed(feed, registry, limit=limit, parser=parser)
        results.append(result)
        articles.extend(result.articles)

    broken = [result for result in results if not result.ok]
    if broken:
        logger.warning("%d of %d feeds failed", len(broken), len(results))
    return articles, results


def attribution_counts(articles: Iterable[Document]) -> dict[str, int]:
    """Articles per company ticker, with unattributed ones counted separately."""
    counts: dict[str, int] = {}
    for article in articles:
        key = str(article.metadata.get("company") or "unattributed")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
