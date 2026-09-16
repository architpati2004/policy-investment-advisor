"""Phase 9 tests: RSS ingestion, recency scoring and the news index.

No network: feeds are injected as parsed structures through the ``parser``
argument, which is what that argument exists for. The two design decisions this
phase turns on — decay applies to news only, and news lives in its own index —
are asserted directly rather than left as comments.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import Settings, get_settings
from backend.exceptions import InvalidFeedsError
from backend.ingestion.company_registry import Company, CompanyRegistry
from backend.ingestion.news import (
    Feed,
    article_document,
    attribution_counts,
    clean_article,
    fetch_all,
    fetch_feed,
    load_feeds,
    parse_published,
)
from backend.rag.retrieval import NewsRetriever, RetrievalPolicy, freshness
from backend.rag.vector_store import SearchResult, VectorIndex, news_index

WORD = re.compile(r"[a-z0-9]+")
TODAY = date(2026, 9, 16)

GODREJ = Company("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer", "gcpl"))
HDFC = Company("HDFCBANK", "HDFC Bank", "Banking")


class FakeEntry(dict):
    """A feedparser entry, which is a dict with attribute access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class FakeParsed:
    """What ``feedparser.parse`` returns."""

    def __init__(self, entries: list[FakeEntry], *, status: int = 200, bozo: bool = False,
                 bozo_exception: Any = None) -> None:
        self.entries = entries
        self.status = status
        self.bozo = bozo
        self.bozo_exception = bozo_exception


def _entry(title: str, summary: str = "", published: date | None = None, link: str = "") -> FakeEntry:
    entry = FakeEntry(title=title, summary=summary, link=link or f"https://example.test/{hash(title)}")
    if published:
        entry["published_parsed"] = published.timetuple()
    return entry


def _parser(parsed: FakeParsed | Exception):
    def _parse(url: str, **kwargs: Any) -> FakeParsed:
        if isinstance(parsed, Exception):
            raise parsed
        return parsed

    return _parse


@pytest.fixture
def registry() -> CompanyRegistry:
    return CompanyRegistry([GODREJ, HDFC])


@pytest.fixture
def feed() -> Feed:
    return Feed(name="Test Wire", url="https://example.test/rss")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={"vectorstore_dir": str(tmp_path / "vectorstore"), "data_dir": str(tmp_path / "data")}
    )


# --- Decision 1: recency applies to news only --------------------------------


def test_news_decays_with_age() -> None:
    weights = [
        freshness({"document_type": "news", "date": (TODAY - timedelta(days=d)).isoformat()},
                  half_life_days=45, floor=0.3, today=TODAY)
        for d in (0, 15, 30, 45, 60)
    ]
    assert weights == sorted(weights, reverse=True), "older must never outweigh newer"
    assert weights[0] == 1.0
    assert weights[3] == pytest.approx(0.5, abs=0.01), "one half-life halves the weight"


def test_a_binding_regulation_never_decays() -> None:
    """The constraint this phase had to satisfy.

    An unamended circular binds exactly as much today as when it was published.
    Ageing it would bury a rule for the crime of being settled law.
    """
    for document_type in ("policy", "circular", "regulation", "budget"):
        weight = freshness(
            {"document_type": document_type, "date": "2015-01-01"},
            half_life_days=45, floor=0.3, today=TODAY,
        )
        assert weight == 1.0, f"{document_type} from 2015 was demoted"


def test_filings_never_decay_either() -> None:
    for document_type in ("annual_report", "quarterly_result", "fundamentals"):
        assert freshness({"document_type": document_type, "date": "2019-03-31"},
                         half_life_days=45, floor=0.3, today=TODAY) == 1.0


def test_decay_has_a_floor_so_age_demotes_but_never_erases() -> None:
    ancient = freshness({"document_type": "news", "date": "2016-01-01"},
                        half_life_days=45, floor=0.3, today=TODAY)
    assert ancient == 0.3
    assert ancient > 0, "a weight of zero would erase the article, not demote it"


def test_an_undated_article_is_not_treated_as_old() -> None:
    """Unknown is not ancient. Feeds omit dates constantly, and defaulting them
    to old would bury exactly the articles with the weakest provenance."""
    assert freshness({"document_type": "news", "date": None},
                     half_life_days=45, floor=0.3, today=TODAY) == 1.0
    assert freshness({"document_type": "news"},
                     half_life_days=45, floor=0.3, today=TODAY) == 1.0


def test_an_unparseable_date_does_not_decay() -> None:
    assert freshness({"document_type": "news", "date": "last Tuesday"},
                     half_life_days=45, floor=0.3, today=TODAY) == 1.0


def test_recency_reorders_but_never_filters(settings: Settings, tmp_path: Path) -> None:
    """Floors are applied to the raw score; decay only sorts.

    An old article that clears the relevance floor must still be returned — lower
    down, but returned. If decay were applied before filtering it could push a
    relevant article under the threshold and out of the results entirely.
    """
    index = VectorIndex(tmp_path / "news_index", FakeEmbeddings(), name="news", settings=settings)
    old = _article("repo rate decision by the central bank", TODAY - timedelta(days=400))
    new = _article("repo rate decision by the central bank", TODAY - timedelta(days=1))
    index.add_documents([old, new])

    retriever = NewsRetriever(index=index, settings=settings,
                              policy=RetrievalPolicy(top_k=5, min_score=0.0, relative_ratio=0.0))
    results = retriever.retrieve("repo rate decision", today=TODAY).results

    assert len(results) == 2, "the older article must still be present"
    assert results[0].metadata["date"] == (TODAY - timedelta(days=1)).isoformat()


def _article(text: str, published: date, company: str = "GODREJCP") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "Test Wire",
            "document_type": "news",
            "title": text[:40],
            "date": published.isoformat(),
            "company": company,
            "sector": "FMCG",
            "url": f"https://example.test/{published.isoformat()}/{abs(hash(text)) % 1000}",
            "chunk_id": f"{published.isoformat()}-{abs(hash(text)) % 10_000}",
        },
    )


# --- Decision 2: news has its own index --------------------------------------


def test_the_news_index_is_separate_from_the_company_index(settings: Settings) -> None:
    """A 589-page filing is 1,776 chunks; an article is one. Sharing a vector
    space lets the filing's mass decide every top-k."""
    news = news_index(FakeEmbeddings(), settings)
    assert news.directory == settings.news_index_dir
    assert news.directory != settings.company_index_dir
    assert news.name == "news"


def test_news_scoping_reuses_the_portfolio_scope(settings: Settings, tmp_path: Path) -> None:
    index = VectorIndex(tmp_path / "news_index", FakeEmbeddings(), name="news", settings=settings)
    index.add_documents(
        [
            _article("godrej consumer raises prices on soaps", TODAY, "GODREJCP"),
            _article("hdfc bank deposit growth accelerates", TODAY, "HDFCBANK"),
        ]
    )
    retriever = NewsRetriever(index=index, settings=settings,
                              policy=RetrievalPolicy(top_k=5, min_score=0.0, relative_ratio=0.0))

    results = retriever.retrieve("growth", companies=["GODREJCP"], today=TODAY).results
    assert {r.metadata["company"] for r in results} == {"GODREJCP"}


# --- Feed parsing ------------------------------------------------------------


def test_a_malformed_feed_that_still_parses_is_used(registry, feed) -> None:
    """RBI's own feed sets bozo and returns ten good entries. Malformed-but-
    parseable is the normal case, so bozo alone must not reject a feed."""
    parsed = FakeParsed(
        [_entry("Repo rate held at 6.50 per cent", "The MPC voted to hold rates steady.", TODAY)],
        bozo=True,
        bozo_exception=ValueError("undeclared entity"),
    )
    result = fetch_feed(feed, registry, parser=_parser(parsed))

    assert result.ok
    assert len(result.articles) == 1


def test_a_feed_that_raises_does_not_abort_the_run(registry, feed) -> None:
    result = fetch_feed(feed, registry, parser=_parser(OSError("connection reset")))

    assert not result.ok
    assert "connection reset" in (result.error or "")
    assert result.articles == []


def test_an_http_error_is_reported_not_raised(registry, feed) -> None:
    result = fetch_feed(feed, registry, parser=_parser(FakeParsed([], status=503)))
    assert not result.ok and "503" in (result.error or "")


def test_an_empty_feed_explains_itself_with_the_bozo_reason(registry, feed) -> None:
    parsed = FakeParsed([], bozo=True, bozo_exception=ValueError("not well-formed"))
    result = fetch_feed(feed, registry, parser=_parser(parsed))

    assert not result.ok
    assert "no entries" in (result.error or "")
    assert "not well-formed" in (result.error or "")


def test_one_broken_feed_does_not_cost_the_others(registry) -> None:
    """Feeds break constantly; a run that stops at the first failure is useless."""
    good = Feed(name="Working", url="https://example.test/ok")
    bad = Feed(name="Broken", url="https://example.test/bad")

    def parser(url: str, **kwargs: Any) -> FakeParsed:
        if url.endswith("bad"):
            raise OSError("name resolution failed")
        return FakeParsed([_entry("Markets rally", "Benchmarks closed higher.", TODAY)])

    articles, results = fetch_all([bad, good, bad], registry, parser=parser)

    assert len(articles) == 1, "the working feed still produced its article"
    assert [r.ok for r in results] == [False, True, False]


def test_a_bad_entry_does_not_lose_the_rest_of_the_feed(registry, feed) -> None:
    class Exploding(FakeEntry):
        def get(self, item: str, default: Any = None) -> Any:
            if item == "title":
                raise RuntimeError("malformed entry")
            return super().get(item, default)

    parsed = FakeParsed([Exploding(), _entry("Good one", "Plenty of text in this article body.", TODAY)])
    result = fetch_feed(feed, registry, parser=_parser(parsed))

    assert len(result.articles) == 1
    assert result.skipped == 1


def test_entries_are_limited(registry, feed) -> None:
    entries = [_entry(f"Story {i}", "Body text long enough to be indexed here.", TODAY) for i in range(20)]
    result = fetch_feed(feed, registry, limit=5, parser=_parser(FakeParsed(entries)))
    assert len(result.articles) == 5


# --- Article construction ----------------------------------------------------


def test_feed_html_is_stripped(registry) -> None:
    assert clean_article("<p>Repo rate <b>held</b> at 6.50%</p>") == "Repo rate held at 6.50%"
    assert clean_article("") == ""


def test_publication_dates_are_normalised() -> None:
    entry = _entry("x", published=date(2026, 8, 10))
    assert parse_published(entry) == "2026-08-10"


def test_a_missing_date_stays_missing(registry, feed) -> None:
    """Stamping an undated article with today would be inventing a fact."""
    assert parse_published(_entry("x")) is None

    document = article_document(
        _entry("Godrej Consumer Products lifts prices", "A long enough body to be indexed."),
        feed,
        registry,
    )
    assert document is not None
    assert "date" not in document.metadata


def test_an_article_records_everything_a_citation_needs(registry, feed) -> None:
    document = article_document(
        _entry(
            "Godrej Consumer Products lifts soap prices",
            "<p>The company raised prices across its home care range.</p>",
            published=date(2026, 9, 10),
            link="https://example.test/story",
        ),
        feed,
        registry,
    )

    assert document is not None
    meta = document.metadata
    assert meta["source"] == "Test Wire"
    assert meta["document_type"] == "news"
    assert meta["date"] == "2026-09-10"
    assert meta["url"] == "https://example.test/story"
    assert meta["company"] == "GODREJCP"
    assert meta["sector"] == "FMCG"
    assert "soap prices" in document.page_content
    assert "home care" in document.page_content


def test_attribution_reuses_the_company_registry(registry, feed) -> None:
    """Not a second mapping mechanism: the same declarations the portfolio and
    the company index already key on."""
    document = article_document(
        _entry("GCPL gains on margin outlook", "Analysts raised targets after the update.", TODAY),
        feed,
        registry,
    )
    assert document is not None and document.metadata["company"] == "GODREJCP"


def test_an_article_naming_two_holdings_records_both(registry, feed) -> None:
    document = article_document(
        _entry(
            "HDFC Bank and Godrej Consumer Products both gain",
            "Both stocks rose in an otherwise flat session for the benchmarks.",
            TODAY,
        ),
        feed,
        registry,
    )

    assert document is not None
    assert document.metadata["company"] in {"GODREJCP", "HDFCBANK"}
    assert set(document.metadata["mentions"].split(",")) == {"GODREJCP", "HDFCBANK"}


def test_an_article_about_nobody_is_still_indexed_without_a_company(registry, feed) -> None:
    """Macro news matters to a portfolio even when it names no holding."""
    document = article_document(
        _entry("RBI holds the repo rate", "The committee voted to keep rates unchanged.", TODAY),
        feed,
        registry,
    )
    assert document is not None
    assert "company" not in document.metadata


def test_a_feed_level_sector_tags_articles_it_cannot_attribute(registry) -> None:
    banking_feed = Feed(name="Bank Wire", url="https://example.test/bank", sector="Banking")
    document = article_document(
        _entry("Deposit growth slows across lenders", "Sector-wide deposit growth eased.", TODAY),
        banking_feed,
        registry,
    )
    assert document is not None and document.metadata["sector"] == "Banking"


def test_a_headline_with_no_body_is_skipped(registry, feed) -> None:
    assert article_document(_entry("Markets", "", TODAY), feed, registry) is None


def test_attribution_counts_separates_the_unattributed() -> None:
    articles = [
        _article("a", TODAY, "GODREJCP"),
        _article("b", TODAY, "GODREJCP"),
        Document(page_content="macro", metadata={"document_type": "news"}),
    ]
    assert attribution_counts(articles) == {"GODREJCP": 2, "unattributed": 1}


# --- Feed list ---------------------------------------------------------------


def test_feeds_load_from_json(tmp_path: Path) -> None:
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps([
        {"name": "RBI", "url": "https://example.test/rbi.xml"},
        {"name": "Bank Wire", "url": "https://example.test/bank.xml", "sector": "Banking"},
    ]))

    feeds = load_feeds(path)
    assert [f.name for f in feeds] == ["RBI", "Bank Wire"]
    assert feeds[1].sector == "Banking"


def test_a_missing_feed_list_is_not_an_error(tmp_path: Path) -> None:
    assert load_feeds(tmp_path / "nothing.json") == []


@pytest.mark.parametrize(
    "payload",
    [
        "{ not json",
        json.dumps({"name": "x"}),
        json.dumps([{"name": "no url"}]),
        json.dumps([{"name": "bad scheme", "url": "ftp://example.test/rss"}]),
    ],
)
def test_malformed_feed_lists_say_what_to_fix(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "feeds.json"
    path.write_text(payload)
    with pytest.raises(InvalidFeedsError) as exc:
        load_feeds(path)
    assert "feeds.json" in str(exc.value)


# --- Indexing ----------------------------------------------------------------


def test_the_same_article_is_not_indexed_twice(settings: Settings, tmp_path: Path) -> None:
    """Feeds repeat their entries on every poll."""
    index = VectorIndex(tmp_path / "news_index", FakeEmbeddings(), name="news", settings=settings)
    article = _article("repo rate held steady", TODAY)

    first = index.add_documents([article])
    second = index.add_documents([article])

    assert (first.added, second.added) == (1, 0)
    assert index.count() == 1


def test_citations_carry_the_publication_date(settings: Settings, tmp_path: Path) -> None:
    index = VectorIndex(tmp_path / "news_index", FakeEmbeddings(), name="news", settings=settings)
    index.add_documents([_article("repo rate held steady", date(2026, 9, 10))])

    retriever = NewsRetriever(index=index, settings=settings,
                              policy=RetrievalPolicy(top_k=5, min_score=0.0, relative_ratio=0.0))
    citation = retriever.retrieve("repo rate", today=TODAY).results[0].citation()

    assert "2026-09-10" in citation
    assert "Test Wire" in citation


def test_a_date_feedparser_rejects_is_still_read() -> None:
    """RBI stamps press releases without a timezone, which RFC 822 requires, so
    feedparser returns None and every RBI article would arrive undated — losing
    exactly the metadata that recency ranking depends on."""
    entry = FakeEntry(title="x", summary="y", published="Wed, 16 Sep 2026 14:30:00",
                      published_parsed=None)
    assert parse_published(entry) == "2026-09-16"


def test_an_unreadable_date_string_is_left_missing() -> None:
    entry = FakeEntry(title="x", summary="y", published="whenever", published_parsed=None)
    assert parse_published(entry) is None
