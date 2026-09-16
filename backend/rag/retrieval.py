"""Deciding which retrieved chunks are worth showing the model.

Phase 4 left ``MIN_RELEVANCE_SCORE`` doing nothing: at 0.25, 69 of 73 chunks
cleared it and ``RETRIEVAL_TOP_K`` was the only real filter. Measuring the real
corpus explained why a single number cannot do this job:

===========================================  =======
question                                      best
===========================================  =======
in-corpus, worst case (CRR amendment)          0.469
out-of-corpus, worst case (NBFC deposits)      0.539
out-of-corpus, clearly unrelated (GST)         0.276
===========================================  =======

A question the corpus *cannot* answer outscores three it can, because the
embedding model measures topical similarity, not applicability: asking about
NBFC deposit rules retrieves commercial-bank deposit rules, which look almost
identical and do not apply. **No absolute threshold separates those two rows.**

So relevance is filtered in two ways here, and refused in a third elsewhere:

1. **An absolute floor** (``MIN_RELEVANCE_SCORE``) rejects questions that are
   simply not about this corpus — GST, income tax, anything under ~0.35. It is a
   property of the embedding model's scale, not of how many documents are
   indexed, so it does not need retuning as the corpus grows.
2. **A relative floor** (``RELATIVE_RELEVANCE_RATIO``) keeps only chunks scoring
   within a fraction of the best match. This is the part that scales: as the
   corpus grows and the best match improves, the bar rises with it
   automatically. A 0.43 chunk is worth reading when the best is 0.47 and worth
   discarding when the best is 0.68.
3. **The prompt** refuses the adjacent case — right topic, wrong institution —
   because no score can. See :mod:`backend.rag.prompts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Literal, Sequence

from backend.config import Settings, get_settings
from backend.logging_config import get_logger
from backend.rag.vector_store import (
    SearchResult,
    VectorIndex,
    company_index,
    news_index,
    policy_index,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalPolicy:
    """How many chunks to fetch and which of them count as relevant."""

    top_k: int
    min_score: float
    relative_ratio: float

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalPolicy:
        settings = settings or get_settings()
        return cls(
            top_k=settings.retrieval_top_k,
            min_score=settings.min_relevance_score,
            relative_ratio=settings.relative_relevance_ratio,
        )

    def floor_for(self, best_score: float) -> float:
        """The effective score floor given the best match in a result set."""
        return max(self.min_score, best_score * self.relative_ratio)


@dataclass(frozen=True)
class Retrieval:
    """What retrieval found, and why it kept or dropped it."""

    question: str
    results: list[SearchResult]
    considered: int
    best_score: float | None
    floor: float | None
    reason: str | None = None

    @property
    def has_context(self) -> bool:
        """True when at least one chunk survived filtering."""
        return bool(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "considered": self.considered,
            "kept": len(self.results),
            "best_score": round(self.best_score, 4) if self.best_score is not None else None,
            "floor": round(self.floor, 4) if self.floor is not None else None,
            "reason": self.reason,
        }


def select_relevant(
    results: list[SearchResult],
    policy: RetrievalPolicy,
) -> tuple[list[SearchResult], float | None, str | None]:
    """Apply the absolute and relative floors to a scored result set.

    Args:
        results: Scored chunks, in any order.
        policy: The floors to apply.

    Returns:
        ``(kept, floor, reason)`` — the surviving chunks best-first, the floor
        that was applied, and, when nothing survived, a sentence explaining why
        that the CLI and API can show the user.
    """
    if not results:
        return [], None, "the index returned no chunks for this question"

    ranked = sorted(results, key=lambda result: result.score, reverse=True)
    best = ranked[0].score

    if best < policy.min_score:
        return (
            [],
            policy.min_score,
            (
                f"the closest chunk scored {best:.2f}, below the "
                f"{policy.min_score:.2f} floor — this corpus is not about that"
            ),
        )

    floor = policy.floor_for(best)
    kept = [result for result in ranked if result.score >= floor]
    logger.debug(
        "Relevance: best %.3f, floor %.3f, kept %d/%d", best, floor, len(kept), len(ranked)
    )
    return kept, floor, None


def scope_filter(
    companies: Sequence[str] | None = None,
    sectors: Sequence[str] | None = None,
) -> Callable[[dict[str, Any]], bool] | None:
    """Build a metadata predicate restricting retrieval to a portfolio's scope.

    A predicate rather than an equality dict because the useful question is
    disjunctive: a Budget change to FMCG taxation matters to a holder of a
    consumer-goods company whether or not that specific company is named, so
    "company in mine **or** sector in mine" is the scope, and a dict of
    equalities can only say *and*.

    Matching is case-insensitive, since a portfolio typed by hand will not agree
    with a registry on capitalisation. ``None`` means no restriction at all.
    """
    wanted_companies = {value.strip().casefold() for value in (companies or []) if value.strip()}
    wanted_sectors = {value.strip().casefold() for value in (sectors or []) if value.strip()}
    if not wanted_companies and not wanted_sectors:
        return None

    def matches(metadata: dict[str, Any]) -> bool:
        company = str(metadata.get("company") or "").strip().casefold()
        sector = str(metadata.get("sector") or "").strip().casefold()
        if bool(company) and company in wanted_companies:
            return True
        if bool(sector) and sector in wanted_sectors:
            return True
        # A news article naming several holdings is attributed to one of them
        # and records the rest here. Free recall: these are registry matches
        # already, so reading them adds no false positives.
        mentions = str(metadata.get("mentions") or "")
        return any(
            ticker.strip().casefold() in wanted_companies
            for ticker in mentions.split(",")
            if ticker.strip()
        )

    return matches


class PolicyRetriever:
    """Retrieves policy chunks and applies the relevance policy to them."""

    def __init__(
        self,
        index: VectorIndex | None = None,
        settings: Settings | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.index = index if index is not None else policy_index(settings=self._settings)
        self.policy = policy or RetrievalPolicy.from_settings(self._settings)

    def retrieve(self, question: str, *, k: int | None = None) -> Retrieval:
        """Fetch and filter the chunks for one question.

        Raises:
            IndexNotFoundError: the policy index has not been built.
        """
        top_k = self.policy.top_k if k is None else k
        # Scored without a floor here; filtering is this module's decision, and
        # the unfiltered scores are what make the "why" message meaningful.
        candidates = self.index.search(question, k=top_k, min_score=0.0)
        kept, floor, reason = select_relevant(candidates, self.policy)

        if reason:
            logger.info("No usable context for %r: %s", question, reason)

        return Retrieval(
            question=question,
            results=kept,
            considered=len(candidates),
            best_score=candidates[0].score if candidates else None,
            floor=floor,
            reason=reason,
        )


class CompanyRetriever:
    """Retrieves company chunks, optionally scoped to a portfolio's holdings.

    Same relevance policy as the policy retriever — the floors are properties of
    the embedding model, not of what is being searched — with the addition that
    a search can be narrowed to particular companies or sectors.
    """

    def __init__(
        self,
        index: VectorIndex | None = None,
        settings: Settings | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.index = index if index is not None else company_index(settings=self._settings)
        self.policy = policy or RetrievalPolicy.from_settings(self._settings)

    def retrieve(
        self,
        question: str,
        *,
        companies: Sequence[str] | None = None,
        sectors: Sequence[str] | None = None,
        k: int | None = None,
    ) -> Retrieval:
        """Fetch and filter company chunks for one question.

        Args:
            question: Natural-language question.
            companies: Tickers to restrict to; empty means no restriction.
            sectors: Sectors to restrict to. Combined with ``companies`` as a
                union, not an intersection — see :func:`scope_filter`.
            k: Chunks to retrieve before filtering; defaults to ``RETRIEVAL_TOP_K``.

        Raises:
            IndexNotFoundError: the company index has not been built.
        """
        top_k = self.policy.top_k if k is None else k
        scope = scope_filter(companies, sectors)
        candidates = self.index.search(question, k=top_k, min_score=0.0, filter=scope)
        kept, floor, reason = select_relevant(candidates, self.policy)

        if reason and scope is not None:
            # Distinguish "nothing relevant" from "nothing in scope", which are
            # different problems with different fixes.
            reason = f"{reason} (searching only {_describe_scope(companies, sectors)})"

        if reason:
            logger.info("No usable company context for %r: %s", question, reason)

        return Retrieval(
            question=question,
            results=kept,
            considered=len(candidates),
            best_score=candidates[0].score if candidates else None,
            floor=floor,
            reason=reason,
        )


def _describe_scope(
    companies: Sequence[str] | None,
    sectors: Sequence[str] | None,
) -> str:
    """Readable rendering of a scope, for the 'why nothing came back' message."""
    parts = []
    if companies:
        parts.append(f"companies {', '.join(companies)}")
    if sectors:
        parts.append(f"sectors {', '.join(sectors)}")
    return " and ".join(parts) if parts else "the whole index"


# --- Merging two corpora -----------------------------------------------------

#: Where a chunk came from. Not cosmetic: an impact assessment has to connect a
#: rule to a holding, so the model must be able to tell which is which.
Origin = Literal["policy", "holding", "news"]


@dataclass(frozen=True)
class ContextChunk:
    """A retrieved chunk, tagged with the index it came from."""

    result: SearchResult
    origin: Origin

    @property
    def score(self) -> float:
        return self.result.score

    @property
    def metadata(self) -> dict[str, Any]:
        return self.result.metadata


@dataclass(frozen=True)
class MergedContext:
    """Context assembled from both indexes, policy first."""

    chunks: list[ContextChunk] = field(default_factory=list)
    policy_reason: str | None = None
    company_reason: str | None = None
    news_reason: str | None = None

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)

    def of(self, origin: Origin) -> list[ContextChunk]:
        return [chunk for chunk in self.chunks if chunk.origin == origin]

    @property
    def reasons(self) -> list[str]:
        """Why a side contributed nothing, for explaining a thin answer."""
        return [
            reason
            for reason in (self.policy_reason, self.company_reason, self.news_reason)
            if reason
        ]


def merge_many(sections: Sequence[tuple[Origin, Retrieval]], limit: int) -> MergedContext:
    """Combine any number of corpora, guaranteeing each a share of the context.

    Generalises the two-corpus merge to the three Phase 10 needs. The reasoning
    is unchanged and still measured: the corpora do not share a score scale, so
    sorting their union by score lets one shut the others out — 5-0 and 0-5
    splits on real questions. Each section is reserved ``limit // n`` slots,
    whatever a section does not use is offered to the others in order, and the
    result is ordered by section so the model reads a rule, then a filing, then
    a story.
    """
    reasons: dict[Origin, str | None] = {origin: retrieval.reason for origin, retrieval in sections}
    if limit < 1 or not sections:
        return MergedContext(
            policy_reason=reasons.get("policy"),
            company_reason=reasons.get("holding"),
            news_reason=reasons.get("news"),
        )

    reserved = max(1, limit // len(sections))
    taken: list[list[SearchResult]] = [
        list(retrieval.results[:reserved]) for _, retrieval in sections
    ]

    # Hand out whatever the reservations left unused, in section order.
    spare = limit - sum(len(group) for group in taken)
    while spare > 0:
        progressed = False
        for index, (_, retrieval) in enumerate(sections):
            if spare <= 0:
                break
            available = retrieval.results[len(taken[index]) :]
            if available:
                taken[index].append(available[0])
                spare -= 1
                progressed = True
        if not progressed:
            break

    chunks: list[ContextChunk] = []
    for (origin, _), group in zip(sections, taken):
        chunks.extend(ContextChunk(result, origin) for result in group)

    logger.info(
        "Merged context: %s (limit %d)",
        ", ".join(f"{origin} {len(group)}" for (origin, _), group in zip(sections, taken)),
        limit,
    )
    return MergedContext(
        chunks=chunks,
        policy_reason=reasons.get("policy"),
        company_reason=reasons.get("holding"),
        news_reason=reasons.get("news"),
    )


def merge_context(
    policy: Retrieval,
    company: Retrieval,
    limit: int,
    news: Retrieval | None = None,
) -> MergedContext:
    """Combine policy, company and (optionally) news chunks under a quota.

    Kept as the two-corpus entry point Phase 8 calls; :func:`merge_many` does
    the work. Merging by raw score does not work here, and the reason is
    measured rather than theoretical: the corpora do not share a score scale.
    Asking "how do the new RBI deposit rate rules affect my holdings?" scores
    policy chunks at 0.49-0.51 and company chunks at 0.40-0.43; asking "what
    regulatory changes affect consumer goods companies?" reverses it. Sorting
    the union by score gave a 5-0 split one way and 0-5 the other — one corpus
    shut the other out every time.
    """
    sections: list[tuple[Origin, Retrieval]] = [("policy", policy), ("holding", company)]
    if news is not None:
        sections.append(("news", news))
    return merge_many(sections, limit)


# --- Recency (news only) -----------------------------------------------------

#: Document types whose relevance does not decay. A circular is a standing
#: instruction: unamended, it binds exactly as much today as when it was
#: published, so ageing it would bury a rule for the crime of being settled law.
TIMELESS_TYPES = frozenset({"policy", "circular", "budget", "regulation",
                            "annual_report", "quarterly_result", "fundamentals", "other"})


def freshness(
    metadata: dict[str, Any],
    *,
    half_life_days: int,
    floor: float,
    today: date | None = None,
) -> float:
    """Rank weight for a chunk, in ``[floor, 1.0]``.

    Recency is a property of the *kind* of document, not of the corpus. Only
    ``news`` decays, because only news makes a claim about a moment; everything
    else returns 1.0 unconditionally, which is what keeps an old-but-binding
    regulation from being buried for being old.

    Two further guarantees:

    * **A floor**, so age demotes an article by at most ``1 - floor`` and never
      erases it.
    * **Undated is not old.** Feeds routinely omit dates, and treating unknown as
      ancient would silently bury exactly the articles whose provenance is
      weakest. No date means no decay.

    This weight is applied to *ordering only*. Relevance floors are applied to
    the raw score, so decay can never push a chunk below a threshold and out of
    the results.
    """
    if str(metadata.get("document_type", "")) in TIMELESS_TYPES:
        return 1.0

    published = metadata.get("date")
    if not published:
        return 1.0
    try:
        published_on = date.fromisoformat(str(published))
    except ValueError:
        return 1.0

    age_days = max((today or date.today()) - published_on, timedelta(0)).days
    if half_life_days <= 0:
        return 1.0
    weight = 0.5 ** (age_days / half_life_days)
    return max(floor, min(1.0, weight))


class NewsRetriever:
    """Retrieves news, ranked by relevance *and* recency.

    Same relevance policy as the other retrievers — the floors belong to the
    embedding model, not to the corpus — with ordering adjusted by
    :func:`freshness` so that, among articles of comparable relevance, the recent
    one leads.
    """

    def __init__(
        self,
        index: VectorIndex | None = None,
        settings: Settings | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.index = index if index is not None else news_index(settings=self._settings)
        self.policy = policy or RetrievalPolicy.from_settings(self._settings)

    def retrieve(
        self,
        question: str,
        *,
        companies: Sequence[str] | None = None,
        sectors: Sequence[str] | None = None,
        k: int | None = None,
        today: date | None = None,
    ) -> Retrieval:
        """Fetch, filter and re-rank news for one question.

        Filtering happens on the raw score and re-ranking after it, in that
        order. Doing it the other way round would let age remove an article
        rather than merely demote it.

        Raises:
            IndexNotFoundError: the news index has not been built.
        """
        top_k = self.policy.top_k if k is None else k
        scope = scope_filter(companies, sectors)
        candidates = self.index.search(question, k=top_k, min_score=0.0, filter=scope)

        kept, floor, reason = select_relevant(candidates, self.policy)

        ranked = sorted(
            kept,
            key=lambda result: result.score
            * freshness(
                result.metadata,
                half_life_days=self._settings.news_half_life_days,
                floor=self._settings.news_freshness_floor,
                today=today,
            ),
            reverse=True,
        )

        if reason and scope is not None:
            reason = f"{reason} (searching only {_describe_scope(companies, sectors)})"
        if reason:
            logger.info("No usable news for %r: %s", question, reason)

        return Retrieval(
            question=question,
            results=ranked,
            considered=len(candidates),
            best_score=candidates[0].score if candidates else None,
            floor=floor,
            reason=reason,
        )
