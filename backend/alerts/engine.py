"""The alert engine: what changed that bears on a holding.

**Alerts are driven by retrieval per holding, not by attribution.** Phase 9
measured company attribution on live feeds at roughly 2.5% — three articles in a
hundred and twenty named any company in the registry, and two of the six named
none at all. An engine triggered by "an article mentions your company" would
therefore be silent most of the time, and its silence would be indistinguishable
from working correctly: it would mean "no headline named you", not "nothing
happened".

So the engine iterates **holdings**, not documents. For each holding it builds a
watch query out of that holding's own identity, retrieves from all three corpora,
and asks the model whether anything in front of it is material. An FMCG holder
gets "input costs are rising across consumer goods" even though no article names
the company — the case attribution can never reach. Iterating holdings also
bounds cost by portfolio size rather than corpus size: one holding is one local
generation, where keying on documents would have meant one per article.

Two rules keep this from degenerating into noise:

* **Grounded or silent.** An alert is raised only when the model names a
  development *and* cites at least one supplied source. Section 11 of CLAUDE.md
  records that no-impact answers cite nothing; here that defect fails safe,
  because an uncited judgement simply produces no alert.
* **Deduplicated on evidence.** The fingerprint is the holding plus the chunks
  actually cited, so a circular seen on ten consecutive runs raises one alert.
  New evidence is a new fingerprint, which is what lets a genuinely new
  development through.
* **A filing cannot be the whole story.** An annual report is a reference
  document: nothing in it is new, and the first live run proved the cost of
  ignoring that, raising an "alert" that the company has audit reports and ESG
  initiatives. Filings are retrieved because they are the best evidence for
  *why* a development matters — the company's own account of its input costs
  behind a story about palm oil prices — but an alert must cite at least one
  policy or news source, something that actually happened.

Feed-level sector tags inform *scoping* here but never constitute an alert on
their own: "this came from a banking feed" is not evidence that it affects your
bank, and treating it as such would manufacture exactly the relevance the Phase 8
prompt forbids the model from inventing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.db import portfolio as portfolio_service
from backend.db.models import Alert
from backend.db.portfolio import HoldingSummary
from backend.logging_config import get_logger
from backend.rag import prompts
from backend.rag.ollama_client import build_chat_model, invoke_messages
from backend.rag.policy_rag import parse_citations
from backend.rag.retrieval import (
    CompanyRetriever,
    ContextChunk,
    MergedContext,
    NewsRetriever,
    Origin,
    PolicyRetriever,
    Retrieval,
    merge_many,
)

logger = get_logger(__name__)

#: Origins that can carry a *development*. A filing is evidence for why one
#: matters; it is not itself news, however recently it was indexed.
DEVELOPMENT_ORIGINS = frozenset({"policy", "news"})


def watch_query(holding: HoldingSummary) -> str:
    """The question asked on a holding's behalf.

    Built from the holding's own identity rather than a fixed phrase, so
    retrieval has something specific to match: the company by name, and the
    sector, because most regulation names an industry and never a company.
    """
    return (
        f"regulatory changes, policy decisions and developments affecting "
        f"{holding.name} ({holding.ticker}) and the {holding.sector} sector"
    )


def fingerprint(ticker: str | None, chunk_ids: list[str]) -> str:
    """Stable identity for an alert: the holding plus the evidence cited.

    Content-hashed rather than time-based, so re-running the engine on an
    unchanged corpus raises nothing new, while a genuinely different source
    produces a different fingerprint and a fresh alert.
    """
    basis = f"{ticker or '-'}|{'|'.join(sorted(chunk_ids))}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Finding:
    """One holding's outcome from a run, alerting or not."""

    ticker: str
    summary: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    raised: bool = False
    duplicate: bool = False
    reason: str | None = None
    considered: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "summary": self.summary,
            "citations": self.citations,
            "raised": self.raised,
            "duplicate": self.duplicate,
            "reason": self.reason,
            "considered": self.considered,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass(frozen=True)
class RunReport:
    """What a whole run did, including the holdings it found nothing for."""

    portfolio: str
    findings: list[Finding] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def raised(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.raised]

    @property
    def quiet(self) -> list[Finding]:
        """Holdings with nothing to report — the expected majority."""
        return [finding for finding in self.findings if not finding.raised]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio,
            "raised": len(self.raised),
            "checked": len(self.findings),
            "findings": [finding.to_dict() for finding in self.findings],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class AlertEngine:
    """Checks each holding against the corpora and records what is material."""

    def __init__(
        self,
        policy_retriever: PolicyRetriever | None = None,
        company_retriever: CompanyRetriever | None = None,
        news_retriever: NewsRetriever | None = None,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.policy_retriever = policy_retriever or PolicyRetriever(settings=self._settings)
        self.company_retriever = company_retriever or CompanyRetriever(settings=self._settings)
        self.news_retriever = news_retriever or NewsRetriever(settings=self._settings)
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = build_chat_model(self._settings)
        return self._chat_model

    def gather(self, holding: HoldingSummary) -> MergedContext:
        """Retrieve everything bearing on one holding, from all three corpora.

        Policy is unscoped because policy chunks carry no company metadata to
        filter on; company and news are scoped to this holding and its sector,
        so another company's filing cannot end up in its alert. A retriever
        whose index has not been built contributes nothing rather than failing
        the run — a portfolio should still be watchable before the news index
        exists.
        """
        sections: list[tuple[Origin, Retrieval]] = []
        query = watch_query(holding)

        for origin, retrieve in (
            ("policy", lambda: self.policy_retriever.retrieve(query)),
            (
                "holding",
                lambda: self.company_retriever.retrieve(
                    query, companies=[holding.ticker], sectors=[holding.sector]
                ),
            ),
            (
                "news",
                lambda: self.news_retriever.retrieve(
                    query, companies=[holding.ticker], sectors=[holding.sector]
                ),
            ),
        ):
            try:
                sections.append((origin, retrieve()))
            except Exception as exc:  # noqa: BLE001 - an unbuilt index is not fatal
                logger.info("No %s context for %s: %s", origin, holding.ticker, exc)

        return merge_many(sections, self._settings.retrieval_top_k)

    def check(self, holding: HoldingSummary, session: Session, portfolio_id: int) -> Finding:
        """Check one holding and record an alert if the evidence supports one."""
        started = time.perf_counter()
        context = self.gather(holding)

        if not context.has_context:
            return Finding(
                ticker=holding.ticker,
                reason="; ".join(context.reasons) or "nothing retrieved",
                elapsed_seconds=time.perf_counter() - started,
            )

        messages = prompts.build_alert_messages(holding, context.chunks)
        reply = invoke_messages(self.chat_model, messages, self._settings).strip()
        elapsed = time.perf_counter() - started

        if prompts.is_no_alert(reply):
            return Finding(
                ticker=holding.ticker,
                reason="the model found nothing material",
                considered=len(context.chunks),
                elapsed_seconds=elapsed,
            )

        cited, invalid = parse_citations(reply, len(context.chunks))
        if invalid:
            logger.warning("Alert cited sources that were not supplied: %s", invalid)

        if not cited:
            # Grounded or silent: an alert nobody can check is worse than none.
            logger.info("Discarding ungrounded alert for %s", holding.ticker)
            return Finding(
                ticker=holding.ticker,
                summary=reply,
                reason="the judgement cited no source, so it was not raised",
                considered=len(context.chunks),
                elapsed_seconds=elapsed,
            )

        chunks = [context.chunks[number - 1] for number in sorted(cited)]

        if not any(chunk.origin in DEVELOPMENT_ORIGINS for chunk in chunks):
            # Everything cited came from the holding's own filings, which report
            # no development — this is where annual-report boilerplate turns
            # into a spurious warning if it is allowed through.
            logger.info("Discarding filing-only alert for %s", holding.ticker)
            return Finding(
                ticker=holding.ticker,
                summary=prompts.strip_alert_markers(reply),
                reason="the only evidence was the company's own filing, which reports nothing new",
                considered=len(context.chunks),
                elapsed_seconds=elapsed,
            )

        citations = [_citation(chunk) for chunk in chunks]
        identity = fingerprint(holding.ticker, [str(c.get("chunk_id") or c["citation"]) for c in citations])

        existing = (
            session.query(Alert)
            .filter(Alert.portfolio_id == portfolio_id, Alert.fingerprint == identity)
            .one_or_none()
        )
        if existing is not None:
            return Finding(
                ticker=holding.ticker,
                summary=existing.summary,
                citations=citations,
                duplicate=True,
                reason="already raised on this evidence",
                considered=len(context.chunks),
                elapsed_seconds=elapsed,
            )

        alert = Alert(
            portfolio_id=portfolio_id,
            ticker=holding.ticker,
            fingerprint=identity,
            summary=prompts.strip_alert_markers(reply),
            evidence=json.dumps(citations),
        )
        session.add(alert)
        session.flush()
        logger.info("Alert raised for %s: %s", holding.ticker, alert.summary[:80])

        return Finding(
            ticker=holding.ticker,
            summary=alert.summary,
            citations=citations,
            raised=True,
            considered=len(context.chunks),
            elapsed_seconds=elapsed,
        )

    def run(
        self,
        session: Session,
        *,
        portfolio_name: str = portfolio_service.DEFAULT_PORTFOLIO,
    ) -> RunReport:
        """Check every holding in a portfolio.

        Raises:
            PortfolioError: the portfolio does not exist.
        """
        started = time.perf_counter()
        summary = portfolio_service.summarise(session, portfolio_name)
        portfolio = portfolio_service.get_portfolio(session, portfolio_name)

        findings = [self.check(holding, session, portfolio.id) for holding in summary.holdings]

        report = RunReport(
            portfolio=portfolio_name,
            findings=findings,
            elapsed_seconds=time.perf_counter() - started,
        )
        logger.info(
            "Alert run over %s: %d raised, %d quiet",
            portfolio_name,
            len(report.raised),
            len(report.quiet),
        )
        return report


def _citation(chunk: ContextChunk) -> dict[str, Any]:
    """Freeze a cited chunk, so an alert stays auditable after the index moves on."""
    metadata = chunk.metadata
    return {
        "citation": chunk.result.citation(),
        "origin": chunk.origin,
        "score": round(chunk.score, 4),
        "chunk_id": metadata.get("chunk_id"),
        "url": metadata.get("url"),
        "date": metadata.get("date"),
    }


def build_engine(settings: Settings | None = None) -> AlertEngine:
    """Construct the engine from configuration. Contacts nothing."""
    return AlertEngine(settings=settings or get_settings())
