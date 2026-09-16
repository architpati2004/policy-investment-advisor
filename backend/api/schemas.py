"""Request and response shapes for the API.

Separate from the SQLAlchemy models on purpose: a table is a storage decision and
a response is a contract, and letting one define the other means every schema
change becomes an API change. It also keeps money out of floats — the database
stores integer paise, and these carry strings, because a JSON number cannot
represent 1180.50 exactly and a portfolio adds these up.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HoldingIn(BaseModel):
    """A position to add, or add to."""

    ticker: str = Field(..., min_length=1, max_length=32, examples=["GODREJCP"])
    quantity: str = Field(..., examples=["150"], description="Shares; fractional allowed")
    average_cost: str = Field(..., examples=["1180.50"], description="Price paid per share, in rupees")
    bought_on: str | None = Field(None, examples=["2026-04-01"], description="YYYY-MM-DD")


class HoldingOut(BaseModel):
    ticker: str
    name: str
    sector: str
    quantity: str
    average_cost: str
    invested: str
    weight_percent: str
    first_bought_on: str | None = None


class PortfolioOut(BaseModel):
    """Cost basis only. There is no price feed, so nothing here is a valuation."""

    name: str
    invested: str
    holdings: list[HoldingOut]
    sector_weights: dict[str, str]


class ScopeOut(BaseModel):
    """What retrieval is restricted to for this portfolio."""

    tickers: list[str]
    sectors: list[str]


class CompanyOut(BaseModel):
    ticker: str
    name: str
    sector: str


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class PortfolioQuestionIn(QuestionIn):
    portfolio: str | None = None


class SourceOut(BaseModel):
    number: int
    citation: str
    score: float
    cited: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerOut(BaseModel):
    """A grounded answer **or** a grounded refusal.

    ``covered=False`` is a result, not an error, and this endpoint returns 200
    for it. Per CLAUDE.md section 11 the refusal text routinely contains the
    finding — qwen3:4b answers "the rules govern commercial banks, so the FMCG
    holding is unaffected" *as* a refusal — so a client that hides the body when
    ``covered`` is false throws away the answer. ``answer`` is always populated;
    ``reason`` carries the model's own account of what is missing.
    """

    question: str
    answer: str
    covered: bool
    grounded: bool
    reason: str | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class AssessmentOut(AnswerOut):
    """A portfolio impact assessment."""

    portfolio: str
    holdings: list[str] = Field(default_factory=list)
    affected: list[str] = Field(default_factory=list)
    unaffected: list[str] = Field(default_factory=list)
    #: False when the model never declared which holdings are affected. Impact is
    #: then unknown, and a client must not present that as "nothing affected".
    impact_declared: bool = False
    context: dict[str, int] = Field(default_factory=dict)


class IndexOut(BaseModel):
    name: str
    built: bool
    vector_count: int = 0
    dimension: int | None = None
    directory: str | None = None
    embedding_model: str | None = None
    updated_at: str | None = None


class SearchHitOut(BaseModel):
    text: str
    score: float
    citation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchOut(BaseModel):
    """Raw retrieval, with the reason when nothing survived the relevance floors."""

    query: str
    index: str
    results: list[SearchHitOut] = Field(default_factory=list)
    considered: int = 0
    reason: str | None = None


class CitationOut(BaseModel):
    citation: str
    origin: str
    score: float | None = None
    chunk_id: str | None = None
    url: str | None = None
    date: str | None = None


class AlertOut(BaseModel):
    id: int
    ticker: str | None
    summary: str
    citations: list[CitationOut] = Field(default_factory=list)
    created_at: str | None = None
    acknowledged: bool = False


class FindingOut(BaseModel):
    """One holding's outcome from an alert run, including a quiet one.

    Quiet holdings are returned rather than omitted: "we checked and found
    nothing" and "we did not check" are different statements, and a client that
    only ever sees raised alerts cannot tell them apart.
    """

    ticker: str
    raised: bool
    duplicate: bool = False
    summary: str | None = None
    reason: str | None = None
    considered: int = 0
    elapsed_seconds: float = 0.0
    citations: list[CitationOut] = Field(default_factory=list)


class AlertRunOut(BaseModel):
    portfolio: str
    raised: int
    checked: int
    findings: list[FindingOut] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class MessageOut(BaseModel):
    message: str
