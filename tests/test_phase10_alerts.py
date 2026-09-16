"""Phase 10 tests: the alert engine.

The design decision this phase turns on — alerts are driven by retrieval per
holding, not by company attribution, because attribution runs at about 2.5% on
live feeds — is asserted directly: an alert must be raisable from a source that
never names the company.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from backend.alerts.engine import AlertEngine, Finding, fingerprint, watch_query
from backend.config import Settings, get_settings
from backend.db import alerts as alert_store
from backend.db import portfolio as service
from backend.db.models import Alert
from backend.db.session import create_db_engine, init_db
from backend.exceptions import IndexNotFoundError, PortfolioError
from backend.ingestion.company_registry import Company as RegistryCompany
from backend.ingestion.company_registry import CompanyRegistry
from backend.rag import prompts
from backend.rag.prompts import NO_ALERT
from backend.rag.retrieval import ContextChunk, Retrieval, merge_many
from backend.rag.vector_store import SearchResult

GODREJ = RegistryCompany("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer",))
HDFC = RegistryCompany("HDFCBANK", "HDFC Bank", "Banking")


# --- Helpers -----------------------------------------------------------------


def _result(score: float, text: str, **metadata: Any) -> SearchResult:
    meta = {"source": "RBI", "title": "a document", "chunk_id": f"c{abs(hash(text)) % 9999}", **metadata}
    return SearchResult(document=Document(page_content=text, metadata=meta), score=score)


def _retrieval(results: list[SearchResult], reason: str | None = None) -> Retrieval:
    return Retrieval(
        question="q",
        results=results,
        considered=len(results),
        best_score=results[0].score if results else None,
        floor=0.35,
        reason=reason,
    )


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


class StubChatModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[tuple[str, str]]] = []

    def stream(self, messages: list[tuple[str, str]]) -> Any:
        self.calls.append(messages)
        return iter([_Chunk(self.reply)])


class StubRetriever:
    def __init__(self, retrieval: Retrieval | Exception) -> None:
        self.retrieval = retrieval
        self.companies: Any = None
        self.sectors: Any = None

    def retrieve(self, question: str, **kwargs: Any) -> Retrieval:
        self.companies = kwargs.get("companies")
        self.sectors = kwargs.get("sectors")
        if isinstance(self.retrieval, Exception):
            raise self.retrieval
        return self.retrieval


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"database_url": f"sqlite:///{tmp_path}/test.db"})


@pytest.fixture
def session(settings: Settings):
    engine = create_db_engine(settings)
    init_db(engine)
    with Session(engine, expire_on_commit=False) as open_session:
        service.sync_companies(open_session, CompanyRegistry([GODREJ, HDFC]))
        service.add_holding(open_session, "GODREJCP", "150", "1180.50")
        yield open_session
    engine.dispose()


def _engine(
    reply: str,
    *,
    policy: list[SearchResult] | Exception | None = None,
    company: list[SearchResult] | None = None,
    news: list[SearchResult] | None = None,
    settings: Settings,
) -> tuple[AlertEngine, StubChatModel, dict[str, StubRetriever]]:
    model = StubChatModel(reply)
    retrievers = {
        "policy": StubRetriever(policy if isinstance(policy, Exception) else _retrieval(policy or [])),
        "company": StubRetriever(_retrieval(company or [])),
        "news": StubRetriever(_retrieval(news or [])),
    }
    engine = AlertEngine(
        policy_retriever=retrievers["policy"],
        company_retriever=retrievers["company"],
        news_retriever=retrievers["news"],
        chat_model=model,
        settings=settings,
    )
    return engine, model, retrievers


def _check(engine: AlertEngine, session: Session, ticker: str = "GODREJCP") -> Finding:
    summary = service.summarise(session)
    holding = next(h for h in summary.holdings if h.ticker == ticker)
    portfolio = service.get_portfolio(session)
    return engine.check(holding, session, portfolio.id)


# --- The decision: retrieval, not attribution --------------------------------


def test_an_alert_can_come_from_a_source_that_never_names_the_company(
    session: Session, settings: Settings
) -> None:
    """The reason attribution is not the trigger.

    Company attribution on live feeds ran at about 2.5% — most articles name no
    company at all. A sector-level development must still be able to raise an
    alert, or the engine is silent for reasons that have nothing to do with
    whether anything happened.
    """
    sector_news = _result(
        0.55,
        "Palm oil and packaging costs are rising sharply across consumer goods manufacturers.",
        document_type="news",
        sector="FMCG",
        date="2026-09-15",
    )
    engine, _, _ = _engine(
        "Input costs are rising across consumer goods, which bears on margins [1].",
        news=[sector_news],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is True
    assert "GODREJCP" not in sector_news.text, "the source must not name the company"
    assert finding.citations[0]["origin"] == "news"


def test_the_watch_query_carries_the_holding_and_its_sector() -> None:
    """Retrieval needs something specific to match, and most regulation names an
    industry rather than a company."""
    holding = type("H", (), {"ticker": "GODREJCP", "name": "Godrej Consumer Products",
                             "sector": "FMCG", "weight": 100})()
    query = watch_query(holding)

    assert "Godrej Consumer Products" in query
    assert "GODREJCP" in query
    assert "FMCG" in query


def test_company_and_news_are_scoped_to_the_holding_but_policy_is_not(
    session: Session, settings: Settings
) -> None:
    engine, _, retrievers = _engine(NO_ALERT, policy=[_result(0.5, "x")], settings=settings)

    _check(engine, session)

    assert retrievers["company"].companies == ["GODREJCP"]
    assert retrievers["news"].sectors == ["FMCG"]
    # Policy chunks carry no company metadata, so scoping them filters everything.
    assert retrievers["policy"].companies is None


def test_each_holding_costs_one_generation(session: Session, settings: Settings) -> None:
    """Cost is bounded by portfolio size, not corpus size."""
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    engine, model, _ = _engine(NO_ALERT, policy=[_result(0.5, "x")], settings=settings)

    report = engine.run(session)

    assert len(report.findings) == 2
    assert len(model.calls) == 2


# --- Grounded or silent ------------------------------------------------------


def test_an_uncited_judgement_raises_nothing(session: Session, settings: Settings) -> None:
    """Section 11 records that no-impact answers cite nothing. Requiring a
    citation makes that defect fail safe: silence, not an unverifiable warning."""
    engine, _, _ = _engine(
        "Costs are rising and this will hurt margins.",  # no [n]
        news=[_result(0.55, "Input costs climb across the sector.", document_type="news")],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is False
    assert "cited no source" in (finding.reason or "")
    assert session.query(Alert).count() == 0


def test_no_alert_is_believed(session: Session, settings: Settings) -> None:
    engine, _, _ = _engine(NO_ALERT, policy=[_result(0.5, "Routine auction result.")],
                           settings=settings)

    finding = _check(engine, session)

    assert finding.raised is False
    assert "nothing material" in (finding.reason or "")


def test_no_alert_is_recognised_wherever_the_model_puts_it() -> None:
    assert prompts.is_no_alert(f"Some preamble\n{NO_ALERT}")
    assert not prompts.is_no_alert("The rules change deposit rates [1].")


def test_a_citation_that_was_never_supplied_does_not_ground_an_alert(
    session: Session, settings: Settings
) -> None:
    engine, _, _ = _engine("Something happened [7].", policy=[_result(0.5, "one source")],
                           settings=settings)

    finding = _check(engine, session)

    assert finding.raised is False
    assert session.query(Alert).count() == 0


def test_nothing_retrieved_means_no_model_call(session: Session, settings: Settings) -> None:
    engine, model, _ = _engine("should never run", settings=settings)

    finding = _check(engine, session)

    assert model.calls == []
    assert finding.raised is False


def test_an_unbuilt_index_does_not_fail_the_run(session: Session, settings: Settings) -> None:
    """A portfolio should be watchable before every index exists."""
    engine, _, _ = _engine(
        "A new circular changes risk weights [1].",
        policy=IndexNotFoundError("policy", "/nowhere"),
        news=[_result(0.55, "Sector costs rise.", document_type="news")],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is True


# --- Deduplication -----------------------------------------------------------


def test_the_same_evidence_raises_one_alert_not_one_per_run(
    session: Session, settings: Settings
) -> None:
    source = _result(0.6, "Risk weights on unsecured retail rise to 125 per cent.")
    engine, _, _ = _engine("Risk weights rise, affecting lending costs [1].",
                           policy=[source], settings=settings)

    first = _check(engine, session)
    second = _check(engine, session)

    assert (first.raised, second.raised) == (True, False)
    assert second.duplicate is True
    assert session.query(Alert).count() == 1


def test_new_evidence_raises_a_new_alert(session: Session, settings: Settings) -> None:
    engine, _, _ = _engine("Risk weights rise [1].",
                           policy=[_result(0.6, "The first circular.")], settings=settings)
    _check(engine, session)

    engine2, _, _ = _engine("A different rule changed [1].",
                            policy=[_result(0.6, "A second, different circular.")],
                            settings=settings)
    second = _check(engine2, session)

    assert second.raised is True
    assert session.query(Alert).count() == 2


def test_fingerprints_ignore_evidence_order() -> None:
    assert fingerprint("GODREJCP", ["a", "b"]) == fingerprint("GODREJCP", ["b", "a"])
    assert fingerprint("GODREJCP", ["a"]) != fingerprint("HDFCBANK", ["a"])


# --- Stored alerts -----------------------------------------------------------


def test_a_raised_alert_is_auditable_after_the_index_moves_on(
    session: Session, settings: Settings
) -> None:
    engine, _, _ = _engine(
        "The circular raises risk weights [1].",
        policy=[_result(0.6, "Risk weights rise.", url="https://example.test/c", date="2026-08-10")],
        settings=settings,
    )
    _check(engine, session)

    alert = session.query(Alert).one()
    assert alert.ticker == "GODREJCP"
    assert json.dumps(alert.to_dict())
    citation = alert.citations[0]
    assert citation["url"] == "https://example.test/c"
    assert citation["date"] == "2026-08-10"
    assert citation["origin"] == "policy"


def test_the_stored_summary_has_no_sentinel_line(session: Session, settings: Settings) -> None:
    engine, _, _ = _engine("Risk weights rise [1].\nAFFECTED: GODREJCP",
                           policy=[_result(0.6, "Risk weights rise.")], settings=settings)
    _check(engine, session)

    assert "AFFECTED" not in session.query(Alert).one().summary


def test_alerts_list_newest_first_and_hide_acknowledged(
    session: Session, settings: Settings
) -> None:
    for text in ("first circular", "second circular"):
        engine, _, _ = _engine("The circular revises risk weights [1].", policy=[_result(0.6, text)],
                               settings=settings)
        _check(engine, session)

    outstanding = alert_store.list_alerts(session)
    assert len(outstanding) == 2

    alert_store.acknowledge(session, outstanding[0].id)
    assert len(alert_store.list_alerts(session)) == 1
    assert len(alert_store.list_alerts(session, include_acknowledged=True)) == 2


def test_acknowledging_keeps_the_fingerprint_so_it_does_not_return(
    session: Session, settings: Settings
) -> None:
    """Acknowledged, not deleted: otherwise the next run raises it again."""
    source = _result(0.6, "Risk weights rise.")
    engine, _, _ = _engine("Risk weights rise [1].", policy=[source], settings=settings)
    _check(engine, session)
    alert_store.acknowledge_all(session)

    engine2, _, _ = _engine("Risk weights rise [1].", policy=[source], settings=settings)
    again = _check(engine2, session)

    assert again.raised is False
    assert again.duplicate is True


def test_clearing_forgets_the_evidence_so_it_can_be_raised_again(
    session: Session, settings: Settings
) -> None:
    source = _result(0.6, "Risk weights rise.")
    engine, _, _ = _engine("Risk weights rise [1].", policy=[source], settings=settings)
    _check(engine, session)

    assert alert_store.clear_alerts(session) == 1

    engine2, _, _ = _engine("Risk weights rise [1].", policy=[source], settings=settings)
    assert _check(engine2, session).raised is True


def test_acknowledging_something_that_is_not_there_says_so(session: Session) -> None:
    with pytest.raises(PortfolioError):
        alert_store.acknowledge(session, 999)


def test_deleting_a_portfolio_takes_its_alerts(session: Session, settings: Settings) -> None:
    engine, _, _ = _engine("Risk weights rise [1].", policy=[_result(0.6, "x")], settings=settings)
    _check(engine, session)

    session.delete(service.get_portfolio(session))
    session.flush()

    assert session.query(Alert).count() == 0


# --- Three-way merge ---------------------------------------------------------


def test_three_corpora_each_get_a_share() -> None:
    merged = merge_many(
        [
            ("policy", _retrieval([_result(0.9, f"p{i}") for i in range(4)])),
            ("holding", _retrieval([_result(0.5, f"h{i}") for i in range(4)])),
            ("news", _retrieval([_result(0.4, f"n{i}") for i in range(4)])),
        ],
        limit=6,
    )

    assert len(merged.chunks) == 6
    assert len(merged.of("policy")) == 2
    assert len(merged.of("holding")) == 2
    assert len(merged.of("news")) == 2, "the lowest-scoring corpus must not be shut out"


def test_an_empty_corpus_gives_its_share_to_the_others() -> None:
    merged = merge_many(
        [
            ("policy", _retrieval([_result(0.9, f"p{i}") for i in range(5)])),
            ("holding", _retrieval([])),
            ("news", _retrieval([_result(0.4, "n0")])),
        ],
        limit=5,
    )

    assert len(merged.chunks) == 5
    assert len(merged.of("policy")) == 4
    assert len(merged.of("news")) == 1


def test_merged_context_reports_every_reason() -> None:
    merged = merge_many(
        [
            ("policy", _retrieval([], "nothing in policy")),
            ("holding", _retrieval([], "nothing in scope")),
            ("news", _retrieval([], "no news")),
        ],
        limit=5,
    )

    assert not merged.has_context
    assert merged.reasons == ["nothing in policy", "nothing in scope", "no news"]


def test_news_chunks_are_labelled_in_the_prompt() -> None:
    chunks = [
        ContextChunk(_result(0.5, "a rule"), "policy"),
        ContextChunk(_result(0.5, "a story", document_type="news", date="2026-09-15"), "news"),
    ]
    rendered = prompts.format_context(chunks)

    assert "[1] POLICY" in rendered
    assert "[2] NEWS" in rendered
    assert "2026-09-15" in rendered, "a dated report must be distinguishable from a standing rule"


# --- Run report --------------------------------------------------------------


def test_a_run_reports_quiet_holdings_too(session: Session, settings: Settings) -> None:
    """Silence is a result, and has to be visible as one."""
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    engine, _, _ = _engine(NO_ALERT, policy=[_result(0.5, "routine")], settings=settings)

    report = engine.run(session)

    assert report.raised == []
    assert [f.ticker for f in report.quiet] == ["GODREJCP", "HDFCBANK"]
    assert json.dumps(report.to_dict())


def test_running_a_portfolio_that_does_not_exist_says_so(
    session: Session, settings: Settings
) -> None:
    engine, _, _ = _engine(NO_ALERT, settings=settings)
    with pytest.raises(PortfolioError):
        engine.run(session, portfolio_name="No Such Portfolio")


def test_a_filing_alone_cannot_raise_an_alert(session: Session, settings: Settings) -> None:
    """What the first live run got wrong.

    Asked what affects GODREJCP, retrieval returned three chunks of the annual
    report and the model duly announced that the company has audit reports and
    ESG initiatives. Nothing in a filing is new, so it can support an alert but
    never constitute one.
    """
    engine, _, _ = _engine(
        "The audit report and ESG initiatives affect financial performance [1][2].",
        company=[
            _result(0.5, "Independent auditor's report on the financial statements.",
                    document_type="annual_report", company="GODREJCP"),
            _result(0.5, "Our ESG initiatives continued through the year.",
                    document_type="annual_report", company="GODREJCP"),
        ],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is False
    assert "reports nothing new" in (finding.reason or "")
    assert session.query(Alert).count() == 0


def test_a_filing_may_support_an_alert_a_development_anchors(
    session: Session, settings: Settings
) -> None:
    """The case filings are retrieved for: the company's own account of why a
    development matters."""
    engine, _, _ = _engine(
        "Palm oil costs are rising [1], and the company reports palm oil is a major input [2].",
        news=[_result(0.55, "Palm oil prices climbed 18% this quarter.", document_type="news")],
        company=[_result(0.5, "Palm oil is among our largest raw material inputs.",
                         document_type="annual_report", company="GODREJCP")],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is True
    assert {c["origin"] for c in finding.citations} == {"news", "holding"}


def test_an_alert_that_is_only_citation_markers_is_discarded(
    session: Session, settings: Settings
) -> None:
    """The literal qwen3:4b output. It decided to raise, cited correctly, and
    summarised the development as "[1], [2]" — nothing a reader could act on."""
    engine, _, _ = _engine(
        "[1], [2]",
        policy=[_result(0.6, "Risk weights rise to 125 per cent.")],
        news=[_result(0.55, "Palm oil prices climb.", document_type="news")],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is False
    assert "only citation markers" in (finding.reason or "")
    assert session.query(Alert).count() == 0


def test_a_summary_with_real_prose_is_kept(session: Session, settings: Settings) -> None:
    engine, _, _ = _engine(
        "Palm oil prices climbed 18% this quarter, squeezing consumer goods margins [1].",
        news=[_result(0.55, "Palm oil prices climb.", document_type="news")],
        settings=settings,
    )

    finding = _check(engine, session)

    assert finding.raised is True
    assert "Palm oil" in (finding.summary or "")


@pytest.mark.parametrize(
    ("summary", "substantive"),
    [
        ("[1], [2]", False),
        ("[1] [2] [3]", False),
        ("   ", False),
        ("See [1].", False),
        ("Risk weights rise [1].", True),  # terse but actionable: not the target
        ("Palm oil costs are rising sharply [1].", True),
        ("The circular raises risk weights on unsecured retail credit [2].", True),
    ],
)
def test_substantive_prose_detection(summary: str, substantive: bool) -> None:
    assert prompts.has_substantive_prose(summary) is substantive
