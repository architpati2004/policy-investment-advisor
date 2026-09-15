"""Portfolio-aware RAG: the assembly the architecture diagram describes.

Derives the companies and sectors a portfolio covers, searches both indexes —
policy unscoped, company narrowed to the holdings — merges the two under a
quota, and asks the model which holdings a change actually bears on.

Two failure modes shape the design, and both are about *not* inventing a link:

* **Crowding.** The corpora do not share a score scale, so merging by score
  shuts one of them out entirely (measured: 5-0 and 0-5 splits on real
  questions). A model that never sees a holding alongside a rule cannot assess
  impact, and will fall back on what it knows about the company from training.
  :func:`backend.rag.retrieval.merge_context` reserves slots for each.
* **Manufactured relevance.** Retrieval always returns *something*. Asked how a
  banking circular affects a consumer-goods holding, the honest answer is
  usually "it does not", and the prompt says so explicitly. An assessment that
  finds an impact every time is worth nothing.

Impact claims are verified the way citations are. The model declares affected
holdings on a closing ``AFFECTED:`` line, and that line is parsed rather than
read out of the prose — scanning for ticker names cannot tell "GODREJCP is
exposed" from "GODREJCP is unaffected", and an early version of this module
reported a holding as affected in an answer stating the opposite. Declared
tickers are then filtered against what is actually held.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.db import portfolio as portfolio_service
from backend.db.portfolio import HoldingSummary, PortfolioSummary
from backend.logging_config import get_logger
from backend.rag import prompts
from backend.rag.ollama_client import build_chat_model, invoke_messages
from backend.rag.policy_rag import Source, parse_citations
from backend.rag.retrieval import (
    CompanyRetriever,
    ContextChunk,
    MergedContext,
    PolicyRetriever,
    merge_context,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PortfolioAssessment:
    """What a policy question means for a specific portfolio."""

    question: str
    portfolio: str
    answer: str
    covered: bool
    holdings: list[str] = field(default_factory=list)
    affected: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    reason: str | None = None
    invalid_citations: list[int] = field(default_factory=list)
    #: False when the model omitted its AFFECTED line: impact is then unknown,
    #: not empty, and callers must not present silence as 'nothing affected'.
    impact_declared: bool = False
    policy_chunks: int = 0
    company_chunks: int = 0
    elapsed_seconds: float = 0.0
    generated: bool = True

    @property
    def cited_sources(self) -> list[Source]:
        return [source for source in self.sources if source.cited]

    @property
    def grounded(self) -> bool:
        """An answer that cites nothing is not an assessment, it is a guess."""
        return self.covered and bool(self.cited_sources)

    @property
    def unaffected(self) -> list[str]:
        """Holdings the model left off its affected list.

        Meaningful only when :attr:`impact_declared` is true. If the model never
        declared a list, every holding is *unknown*, and calling them unaffected
        would be inventing reassurance.
        """
        if not self.impact_declared:
            return []
        return [ticker for ticker in self.holdings if ticker not in self.affected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "portfolio": self.portfolio,
            "answer": self.answer,
            "covered": self.covered,
            "grounded": self.grounded,
            "holdings": self.holdings,
            "affected": self.affected,
            "unaffected": self.unaffected if self.impact_declared else [],
            "impact_declared": self.impact_declared,
            "reason": self.reason,
            "sources": [source.to_dict() for source in self.sources],
            "invalid_citations": self.invalid_citations,
            "context": {"policy": self.policy_chunks, "company": self.company_chunks},
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


#: The model's closing ``AFFECTED: ...`` line.
_AFFECTED_LINE = re.compile(
    rf"^\s*{re.escape(prompts.AFFECTED_PREFIX)}\s*(?P<tickers>.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def find_affected(answer: str, holdings: list[str]) -> tuple[list[str], bool]:
    """Read the declared impact line, verified against what is actually held.

    Scanning the prose for ticker names does not work, and the failure is not
    subtle: "GODREJCP is exposed to this" and "GODREJCP is unaffected by this"
    both name the ticker, so a free scan reported a holding as affected in an
    answer that said the opposite — a summary line contradicting the text
    beneath it. The model therefore declares the list on its own line, the way
    it declares a refusal, and that line is parsed rather than interpreted.

    Tickers are then filtered against the portfolio: one the user does not hold
    cannot be affected whatever the model writes, and an unverified list would
    put positions in an impact report that do not exist.

    Returns:
        ``(affected, declared)`` — the verified tickers, and whether the model
        actually gave the line. Without it, impact is unknown rather than empty.
    """
    match = None
    for match in _AFFECTED_LINE.finditer(answer):
        pass  # the last one wins, in case the model restates it
    if match is None:
        return [], False

    body = match.group("tickers").strip().strip(".")
    if not body or body.lower() in {"none", "no holdings", "-", "n/a"}:
        return [], True

    held = {ticker.upper(): ticker for ticker in holdings}
    declared = [part.strip().strip("[]()*").upper() for part in body.split(",")]
    affected = [held[name] for name in declared if name in held]

    unknown = [name for name in declared if name and name not in held]
    if unknown:
        logger.warning("Assessment named holdings that are not in the portfolio: %s", unknown)
    return affected, True


def _sources(chunks: list[ContextChunk], cited: set[int]) -> list[Source]:
    return [
        Source(
            number=number,
            citation=chunk.result.citation(),
            score=chunk.score,
            cited=number in cited,
            text=chunk.result.text,
            metadata={**chunk.metadata, "origin": chunk.origin},
        )
        for number, chunk in enumerate(chunks, 1)
    ]


class PortfolioRAG:
    """Answers "what does this mean for my portfolio", or declines to."""

    def __init__(
        self,
        policy_retriever: PolicyRetriever | None = None,
        company_retriever: CompanyRetriever | None = None,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.policy_retriever = policy_retriever or PolicyRetriever(settings=self._settings)
        self.company_retriever = company_retriever or CompanyRetriever(settings=self._settings)
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = build_chat_model(self._settings)
        return self._chat_model

    def gather(self, question: str, summary: PortfolioSummary) -> MergedContext:
        """Retrieve from both indexes and merge under a quota.

        Policy retrieval is unscoped because policy chunks carry no company or
        sector metadata to filter on — a circular is about an industry, and the
        question carries the topic. Company retrieval is narrowed to the
        holdings, which is what keeps another company's annual report out of an
        assessment of this portfolio.
        """
        tickers = [holding.ticker for holding in summary.holdings]
        sectors = sorted({holding.sector for holding in summary.holdings})

        policy = self.policy_retriever.retrieve(question)
        company = self.company_retriever.retrieve(question, companies=tickers, sectors=sectors)
        return merge_context(policy, company, self._settings.retrieval_top_k)

    def assess(
        self,
        question: str,
        session: Session,
        *,
        portfolio_name: str = portfolio_service.DEFAULT_PORTFOLIO,
    ) -> PortfolioAssessment:
        """Assess one question against a portfolio's holdings.

        Raises:
            PortfolioError: the portfolio does not exist.
            IndexNotFoundError: an index has not been built.
            OllamaUnavailableError, LLMTimeoutError: generation failed.
        """
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        summary = portfolio_service.summarise(session, portfolio_name)
        tickers = [holding.ticker for holding in summary.holdings]
        context = self.gather(question, summary)

        if not context.has_context:
            return self._uncovered(question, portfolio_name, tickers, context, started)

        messages = prompts.build_portfolio_messages(question, summary.holdings, context.chunks)
        logger.info(
            "Assessing %r against %d holdings from %d policy + %d company chunks",
            question,
            len(tickers),
            len(context.of("policy")),
            len(context.of("holding")),
        )
        reply = invoke_messages(self.chat_model, messages, self._settings).strip()
        elapsed = time.perf_counter() - started

        policy_count, company_count = len(context.of("policy")), len(context.of("holding"))

        if prompts.is_refusal(reply):
            detail = prompts.refusal_detail(reply)
            logger.info("Model declined to assess: %s", detail)
            return PortfolioAssessment(
                question=question,
                portfolio=portfolio_name,
                answer=reply,
                covered=False,
                holdings=tickers,
                sources=_sources(context.chunks, set()),
                reason=detail,
                policy_chunks=policy_count,
                company_chunks=company_count,
                elapsed_seconds=elapsed,
            )

        cited, invalid = parse_citations(reply, len(context.chunks))
        if invalid:
            logger.warning("Assessment cited sources that were not supplied: %s", invalid)

        affected, declared = find_affected(reply, tickers)
        if not declared and tickers:
            logger.warning("Assessment gave no %s line; impact is unknown", prompts.AFFECTED_PREFIX)
        return PortfolioAssessment(
            question=question,
            portfolio=portfolio_name,
            answer=reply,
            covered=True,
            holdings=tickers,
            affected=affected,
            impact_declared=declared,
            sources=_sources(context.chunks, cited),
            invalid_citations=invalid,
            policy_chunks=policy_count,
            company_chunks=company_count,
            elapsed_seconds=elapsed,
        )

    def _uncovered(
        self,
        question: str,
        portfolio_name: str,
        tickers: list[str],
        context: MergedContext,
        started: float,
    ) -> PortfolioAssessment:
        """Refuse before generating: neither index offered anything to reason from."""
        reason = "; ".join(context.reasons) or "no relevant documents were retrieved"
        return PortfolioAssessment(
            question=question,
            portfolio=portfolio_name,
            answer=f"{prompts.NOT_COVERED}: {reason}",
            covered=False,
            holdings=tickers,
            reason=reason,
            elapsed_seconds=time.perf_counter() - started,
            generated=False,
        )


def build_portfolio_rag(settings: Settings | None = None) -> PortfolioRAG:
    """Construct the chain from configuration. Contacts nothing."""
    return PortfolioRAG(settings=settings or get_settings())
