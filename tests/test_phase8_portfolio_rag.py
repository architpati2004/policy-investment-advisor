"""Phase 8 tests: context merging and portfolio-aware assessment.

Fast tests use stubs for both retrieval and generation. The live tests at the
bottom are marked ``slow``; they run real generation and are excluded unless you
ask with ``pytest -m slow``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import Chunk, StubChatModel
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.db import portfolio as service
from backend.db.session import create_db_engine, init_db
from backend.ingestion.company_registry import Company as RegistryCompany
from backend.ingestion.company_registry import CompanyRegistry
from backend.rag import ollama_client, prompts
from backend.rag.portfolio_rag import PortfolioRAG, find_affected
from backend.rag.prompts import NOT_COVERED
from backend.rag.retrieval import ContextChunk, Retrieval, merge_context
from backend.rag.vector_store import SearchResult

GODREJ = RegistryCompany("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer",))
HDFC = RegistryCompany("HDFCBANK", "HDFC Bank", "Banking")


# --- Helpers -----------------------------------------------------------------


def _policy_result(score: float, text: str = "The repo rate remains unchanged.") -> SearchResult:
    return SearchResult(
        document=Document(
            page_content=text,
            metadata={
                "source": "RBI",
                "title": "RBI_commercial_banks_deposit_rate_directions_2025",
                "page": 6,
                "date": "2025-11-28",
            },
        ),
        score=score,
    )


def _company_result(score: float, ticker: str = "GODREJCP", text: str = "Revenue grew.") -> SearchResult:
    return SearchResult(
        document=Document(
            page_content=text,
            metadata={
                "source": "Godrej Consumer Products",
                "title": f"{ticker}_annual_report_2025",
                "company": ticker,
                "sector": "FMCG",
                "page": 100,
            },
        ),
        score=score,
    )


def _retrieval(results: list[SearchResult], reason: str | None = None) -> Retrieval:
    return Retrieval(
        question="q",
        results=results,
        considered=len(results),
        best_score=results[0].score if results else None,
        floor=0.35,
        reason=reason,
    )


class StubRetriever:
    """Stands in for either retriever, recording how it was scoped."""

    def __init__(self, retrieval: Retrieval) -> None:
        self.retrieval = retrieval
        self.companies: Any = None
        self.sectors: Any = None

    def retrieve(self, question: str, **kwargs: Any) -> Retrieval:
        self.companies = kwargs.get("companies")
        self.sectors = kwargs.get("sectors")
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


def _rag(
    reply: str,
    policy: list[SearchResult],
    company: list[SearchResult],
    settings: Settings,
    policy_reason: str | None = None,
    company_reason: str | None = None,
) -> tuple[PortfolioRAG, StubChatModel, StubRetriever, StubRetriever]:
    model = StubChatModel(reply)
    policy_retriever = StubRetriever(_retrieval(policy, policy_reason))
    company_retriever = StubRetriever(_retrieval(company, company_reason))
    rag = PortfolioRAG(
        policy_retriever=policy_retriever,
        company_retriever=company_retriever,
        chat_model=model,
        settings=settings,
    )
    return rag, model, policy_retriever, company_retriever


# --- Merging -----------------------------------------------------------------


def test_merging_guarantees_each_corpus_a_share() -> None:
    """The measured failure: sorting the union by score gave 5-0 and 0-5 splits.

    Policy chunks here outscore every company chunk, which is exactly the case
    where a score-ordered merge would leave the model with no holding to assess.
    """
    policy = _retrieval([_policy_result(0.51), _policy_result(0.50), _policy_result(0.49)])
    company = _retrieval([_company_result(0.43), _company_result(0.42), _company_result(0.40)])

    merged = merge_context(policy, company, limit=5)

    assert len(merged.chunks) == 5, "an odd slot should be used, not wasted"
    assert len(merged.of("holding")) >= 2, "company chunks must not be crowded out by score"
    assert len(merged.of("policy")) >= 2
    assert all(isinstance(chunk, ContextChunk) for chunk in merged.chunks)


def test_an_unused_reservation_goes_to_the_other_side() -> None:
    """A policy-only question must not waste half its context on an empty half."""
    policy = _retrieval([_policy_result(0.60 - i / 100) for i in range(5)])
    company = _retrieval([])

    merged = merge_context(policy, company, limit=5)

    assert len(merged.of("policy")) == 5
    assert merged.of("holding") == []


def test_a_portfolio_only_question_fills_from_company_chunks() -> None:
    merged = merge_context(_retrieval([]), _retrieval([_company_result(0.45)] * 4), limit=5)
    assert len(merged.of("holding")) == 4
    assert merged.of("policy") == []


def test_policy_context_comes_first() -> None:
    """The rule, then the holding it might bear on: the order the question follows."""
    merged = merge_context(
        _retrieval([_policy_result(0.40)]), _retrieval([_company_result(0.90)]), limit=4
    )
    assert [chunk.origin for chunk in merged.chunks] == ["policy", "holding"]


def test_merging_nothing_reports_both_reasons() -> None:
    merged = merge_context(
        _retrieval([], "nothing in the policy index"),
        _retrieval([], "nothing in scope"),
        limit=5,
    )
    assert not merged.has_context
    assert merged.reasons == ["nothing in the policy index", "nothing in scope"]


def test_a_zero_limit_returns_nothing_rather_than_erroring() -> None:
    merged = merge_context(_retrieval([_policy_result(0.5)]), _retrieval([]), limit=0)
    assert merged.chunks == []


# --- Prompt ------------------------------------------------------------------


def test_the_prompt_shows_the_portfolio(session: Session) -> None:
    summary = service.summarise(session)
    rendered = prompts.format_holdings(summary.holdings)

    assert "GODREJCP" in rendered
    assert "Godrej Consumer Products" in rendered
    assert "FMCG" in rendered  # a rule naming only an industry still has to land
    assert "100.00%" in rendered


def test_sources_are_labelled_by_which_index_they_came_from() -> None:
    chunks = [
        ContextChunk(_policy_result(0.5), "policy"),
        ContextChunk(_company_result(0.4), "holding"),
    ]
    rendered = prompts.format_context(chunks)

    assert "[1] POLICY" in rendered
    assert "[2] HOLDING GODREJCP" in rendered


def test_the_prompt_covers_the_case_where_nothing_bears_on_a_holding() -> None:
    """Retrieval always returns something, so the prompt must handle 'no impact'.

    Asserted on the requirement rather than on exact wording: this prompt went
    through four revisions against qwen3:4b, and a test pinned to one phrasing
    fails on rewording while saying nothing about whether the rule survived.
    """
    text = prompts.PORTFOLIO_SYSTEM_PROMPT.lower()
    assert "nothing in the sources bears on a holding" in text
    assert "rather than inferring" in text, "must forbid manufacturing a connection"
    assert NOT_COVERED in prompts.PORTFOLIO_SYSTEM_PROMPT
    assert prompts.AFFECTED_PREFIX in prompts.PORTFOLIO_SYSTEM_PROMPT


def test_the_portfolio_prompt_names_no_specific_company() -> None:
    text = prompts.PORTFOLIO_SYSTEM_PROMPT.lower()
    assert "godrej" not in text and "fmcg" not in text


# --- Affected holdings -------------------------------------------------------


def test_affected_holdings_come_from_the_declared_line() -> None:
    affected, declared = find_affected(
        "The rules raise costs for banks [1].\nAFFECTED: HDFCBANK", ["GODREJCP", "HDFCBANK"]
    )
    assert (affected, declared) == (["HDFCBANK"], True)


def test_an_unaffected_holding_is_not_reported_as_affected() -> None:
    """The bug this line exists to prevent.

    4b answered "The new commercial bank deposit rate rules do not affect
    GODREJCP holdings" and the old prose scan reported GODREJCP as affected,
    because the sentence names it. The summary line contradicted the assessment
    printed directly above it.
    """
    answer = "The rules do not affect GODREJCP holdings [1].\nAFFECTED: none"
    assert find_affected(answer, ["GODREJCP"]) == ([], True)


def test_a_ticker_that_is_not_held_cannot_be_affected() -> None:
    """Whatever the model writes, a position the user does not hold is not theirs."""
    assert find_affected("AFFECTED: INFY, TCS", ["GODREJCP"]) == ([], True)


def test_declared_tickers_are_matched_case_insensitively() -> None:
    assert find_affected("AFFECTED: godrejcp", ["GODREJCP"]) == (["GODREJCP"], True)


def test_a_missing_line_means_unknown_not_none() -> None:
    """Silence must not be presented to the user as 'nothing is affected'."""
    assert find_affected("Some answer with no declaration [1].", ["GODREJCP"]) == ([], False)


def test_a_restated_line_takes_the_last_one() -> None:
    answer = "AFFECTED: GODREJCP\nOn reflection:\nAFFECTED: none"
    assert find_affected(answer, ["GODREJCP"]) == ([], True)


# --- The chain ---------------------------------------------------------------


def test_company_retrieval_is_scoped_to_the_holdings(session: Session, settings: Settings) -> None:
    rag, _, policy_retriever, company_retriever = _rag(
        "GODREJCP is unaffected [1].\nAFFECTED: none", [_policy_result(0.5)], [_company_result(0.4)], settings
    )

    rag.assess("does this affect me?", session)

    assert company_retriever.companies == ["GODREJCP"]
    assert company_retriever.sectors == ["FMCG"]
    # Policy chunks carry no company or sector metadata, so scoping them would
    # filter everything out.
    assert policy_retriever.companies is None


def test_an_assessment_reports_affected_and_unaffected_holdings(
    session: Session, settings: Settings
) -> None:
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    rag, _, _, _ = _rag(
        "The deposit rules apply to HDFCBANK [1]. GODREJCP is not a bank and is unaffected [2].\n"
        "AFFECTED: HDFCBANK",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("do the deposit rules affect my holdings?", session)

    assert assessment.covered and assessment.grounded
    assert assessment.affected == ["HDFCBANK"]
    assert assessment.unaffected == ["GODREJCP"], "named as unaffected, so not affected"
    assert sorted(assessment.holdings) == ["GODREJCP", "HDFCBANK"]


def test_an_assessment_finding_no_impact_is_still_a_real_answer(
    session: Session, settings: Settings
) -> None:
    """'Nothing here touches your portfolio' is the useful answer much of the time."""
    rag, _, _, _ = _rag(
        "The circular governs commercial banks only, so the portfolio is unaffected [1].\n"
        "AFFECTED: none",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does the deposit circular affect me?", session)

    assert assessment.covered is True
    assert assessment.affected == []
    assert assessment.unaffected == ["GODREJCP"]


def test_both_corpora_reach_the_model(session: Session, settings: Settings) -> None:
    rag, model, _, _ = _rag(
        "GODREJCP unaffected [1].\nAFFECTED: none", [_policy_result(0.5)], [_company_result(0.4)], settings
    )

    rag.assess("anything?", session)
    _, human = model.calls[0]

    assert "POLICY" in human[1] and "HOLDING GODREJCP" in human[1]
    assert "Portfolio:" in human[1]


def test_refusal_before_the_model_when_neither_index_offers_anything(
    session: Session, settings: Settings
) -> None:
    rag, model, _, _ = _rag(
        "should never run", [], [], settings, policy_reason="scored 0.21", company_reason="nothing in scope"
    )

    assessment = rag.assess("what is the GST rate on cement?", session)

    assert model.calls == []
    assert assessment.covered is False
    assert assessment.generated is False
    assert "scored 0.21" in (assessment.reason or "")
    assert "nothing in scope" in (assessment.reason or "")


def test_a_model_refusal_is_believed(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag(
        f"{NOT_COVERED}: the sources say nothing about consumer goods taxation",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("how will GST changes hit my portfolio?", session)

    assert assessment.covered is False
    assert assessment.generated is True
    assert assessment.affected == []
    assert assessment.cited_sources == []


def test_invented_citations_are_flagged(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag("GODREJCP is exposed [9].", [_policy_result(0.5)], [_company_result(0.4)], settings)

    assessment = rag.assess("anything?", session)

    assert assessment.invalid_citations == [9]
    assert assessment.grounded is False


def test_an_empty_portfolio_still_answers_from_policy(
    session: Session, settings: Settings
) -> None:
    """Nothing held means nothing to scope by, not a crash."""
    service.remove_holding(session, "GODREJCP")
    rag, _, _, company_retriever = _rag(
        "The circular raises risk weights [1].", [_policy_result(0.5)], [], settings
    )

    assessment = rag.assess("what changed?", session)

    assert assessment.covered is True
    assert assessment.holdings == []
    assert company_retriever.companies == []


def test_assessment_serialises_for_the_api(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag("GODREJCP unaffected [1].\nAFFECTED: none", [_policy_result(0.5)], [_company_result(0.4)], settings)

    payload = rag.assess("anything?", session).to_dict()

    assert json.dumps(payload)
    assert payload["context"] == {"policy": 1, "company": 1}
    assert payload["sources"][0]["metadata"]["origin"] == "policy"
    assert payload["sources"][1]["metadata"]["origin"] == "holding"


def test_an_empty_question_is_rejected(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag("x", [_policy_result(0.5)], [], settings)
    with pytest.raises(ValueError):
        rag.assess("   ", session)


# --- Live (opt in with `pytest -m slow`) -------------------------------------


def _live_ready() -> bool:
    try:
        from backend.rag.vector_store import company_index, policy_index

        return (
            ollama_client.check_status().ready
            and policy_index().exists
            and company_index().exists
        )
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(
    not _live_ready(), reason="needs Ollama and both indexes built"
)


@pytest.mark.slow
@pytest.mark.integration
@live
def test_live_does_not_manufacture_an_impact(session: Session) -> None:
    """The portfolio-level version of the refusal discipline.

    The corpus is RBI banking regulation; the portfolio holds a consumer-goods
    company. Retrieval will return confident-looking bank chunks regardless, and
    the honest outcome is that the holding is unaffected — or that the sources
    do not cover it. Inventing a mechanism by which deposit-rate rules hit an
    FMCG business is the failure this phase has to avoid.
    """
    from backend.rag.portfolio_rag import build_portfolio_rag

    assessment = build_portfolio_rag().assess(
        "do the new commercial bank deposit rate rules affect my holdings?", session
    )

    if assessment.covered:
        claims_impact = assessment.affected == ["GODREJCP"] and not re.search(
            r"\b(unaffected|not affected|no (direct )?impact|does not apply|no effect)\b",
            assessment.answer,
            re.IGNORECASE,
        )
        assert not claims_impact, (
            "claimed a bank deposit rule affects an FMCG holding without qualification: "
            f"{assessment.answer!r}"
        )
    else:
        assert assessment.reason


def test_a_refusal_is_recognised_wherever_the_model_puts_it() -> None:
    """qwen3:1.7b emitted the AFFECTED line before the refusal, and a
    start-of-reply check read the refusal as an answer."""
    answer = "AFFECTED: none\nNOT_COVERED: the sources are about banks, not this company"
    assert prompts.is_refusal(answer)
    assert prompts.refusal_detail(answer) == "the sources are about banks, not this company"


def test_a_refusal_out_of_order_still_refuses_in_the_chain(
    session: Session, settings: Settings
) -> None:
    rag, _, _, _ = _rag(
        "AFFECTED: none\nNOT_COVERED: nothing here bears on the holding",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does this affect me?", session)

    assert assessment.covered is False, "a refusal must never be read as an assessment"
    assert assessment.affected == []
    assert assessment.reason == "nothing here bears on the holding"


# --- Sentinels stay out of the text a reader sees ----------------------------


def test_the_affected_line_does_not_leak_into_the_answer(
    session: Session, settings: Settings
) -> None:
    """The leak this test exists for.

    The chat page rendered "AFFECTED: none" as the first line of the answer and
    then showed the parsed impact below it, so the machinery was visible twice —
    once as raw text a reader has to know how to ignore.
    """
    rag, _, _, _ = _rag(
        "The rules govern commercial banks, so the holding is unaffected [1].\nAFFECTED: none",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does the deposit circular affect me?", session)

    assert "AFFECTED" not in assessment.answer
    assert assessment.answer.startswith("The rules govern commercial banks")
    # Parsed, not discarded: the meaning moved into the structured fields.
    assert assessment.affected == []
    assert assessment.impact_declared is True


def test_a_declared_impact_is_parsed_before_the_line_is_stripped(
    session: Session, settings: Settings
) -> None:
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    rag, _, _, _ = _rag(
        "Risk weights rise for lenders [1].\nAFFECTED: HDFCBANK",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does this affect me?", session)

    assert assessment.affected == ["HDFCBANK"]
    assert "AFFECTED" not in assessment.answer


def test_a_portfolio_refusal_is_prose_too(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag(
        f"{NOT_COVERED}: the sources say nothing about consumer goods taxation",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("how will GST changes hit my portfolio?", session)

    assert NOT_COVERED not in assessment.answer
    assert assessment.answer == "the sources say nothing about consumer goods taxation"


def test_an_inline_affected_declaration_is_parsed_and_then_stripped(
    session: Session, settings: Settings
) -> None:
    """The leak, in the form that actually reaches the page.

    qwen3 appends the declaration to a sentence as often as it puts it on its own
    line. Parsing only the anchored form would miss it, and stripping only the
    anchored form would show it to the reader — the first version of this fix did
    the second.
    """
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    rag, _, _, _ = _rag(
        "Risk weights rise for lenders [1]. AFFECTED: HDFCBANK",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does this affect me?", session)

    assert "AFFECTED" not in assessment.answer
    assert assessment.answer == "Risk weights rise for lenders [1]."
    assert assessment.affected == ["HDFCBANK"]
    assert assessment.impact_declared is True


def test_an_inline_refusal_keeps_its_reason_clean(session: Session, settings: Settings) -> None:
    rag, _, _, _ = _rag(
        "NOT_COVERED: The sources do not specify the impact on Godrej. AFFECTED: none",
        [_policy_result(0.5)],
        [_company_result(0.4)],
        settings,
    )

    assessment = rag.assess("does this affect me?", session)

    assert assessment.covered is False
    assert "AFFECTED" not in (assessment.reason or "")
    assert "AFFECTED" not in assessment.answer
    assert assessment.reason == "The sources do not specify the impact on Godrej."
