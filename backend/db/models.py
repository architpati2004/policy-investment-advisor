"""Portfolio schema.

Three tables, shaped so that holding a new company is *data*, never a schema
change: companies are rows keyed by ticker, sectors are plain strings rather
than an enum, and nothing anywhere names a specific company. Adding HDFCBANK or
RELIANCE alongside GODREJCP is two inserts.

**Money and quantity are stored as integers.** Floating point is the wrong type
for money — 0.1 + 0.2 is famously not 0.3, and a portfolio multiplies and sums
these values constantly — and SQLite has no decimal type to fall back on, so
SQLAlchemy would quietly round-trip ``Numeric`` through a float anyway. Storing
paise and thousandths of a share keeps every stored value exact; the ``quantity``,
``average_cost`` and ``invested`` properties hand back ``Decimal`` for display
and arithmetic, so callers never see the integers.

The ``Company`` table mirrors ``data/companies/registry.json``, which stays the
source of truth: :func:`backend.db.portfolio.sync_companies` imports it. The
mirror exists because a holding needs something to point at, and because Phase 8
derives "which companies and sectors do I hold" from a query rather than by
reading a JSON file back at retrieval time.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
import json
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Money is stored in paise, quantity in thousandths of a share.
PAISE_PER_RUPEE = Decimal(100)
MILLI_PER_UNIT = Decimal(1000)


class Base(DeclarativeBase):
    """Declarative base for every model in the application."""


def to_paise(rupees: Decimal | float | int | str) -> int:
    """Convert rupees to whole paise, rounding half up.

    Takes ``str`` or ``Decimal`` exactly; a ``float`` is converted through its
    string form, so ``1180.5`` means 1180.50 and not 1180.4999999999999.
    """
    amount = Decimal(str(rupees))
    return int((amount * PAISE_PER_RUPEE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def from_paise(paise: int) -> Decimal:
    """Convert whole paise back to rupees, to two decimal places."""
    return (Decimal(paise) / PAISE_PER_RUPEE).quantize(Decimal("0.01"))


def to_milli(quantity: Decimal | float | int | str) -> int:
    """Convert a share count to thousandths, so fractional units stay exact."""
    amount = Decimal(str(quantity))
    return int((amount * MILLI_PER_UNIT).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def from_milli(milli: int) -> Decimal:
    """Convert thousandths of a share back to a share count."""
    return (Decimal(milli) / MILLI_PER_UNIT).quantize(Decimal("0.001"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    """A company the portfolio can hold, mirrored from the registry.

    Keyed by ticker because that is the identity that does not drift — the same
    key the company index writes into every chunk's metadata, which is what lets
    Phase 8 join a holding to the documents about it.
    """

    __tablename__ = "companies"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    #: Free text, deliberately not an enum: a new sector must not need a migration.
    sector: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    holdings: Mapped[list[Holding]] = relationship(back_populates="company")

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Company(ticker={self.ticker!r}, sector={self.sector!r})"


class Portfolio(Base):
    """A named set of holdings.

    Plural from the start because the alternative — assuming one portfolio and
    hard-coding it — is precisely the kind of assumption that later costs a
    migration. Callers that only ever want one can use the default.
    """

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    holdings: Mapped[list[Holding]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        order_by="Holding.ticker",
    )

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Portfolio(name={self.name!r}, holdings={len(self.holdings)})"


class Holding(Base):
    """One company's position within one portfolio.

    One row per company rather than one per purchase lot: buying more of
    something already held updates the weighted average cost, which is what a
    holdings statement shows and all this system needs. Per-lot tax accounting
    would be a different table, and is not what a policy-impact tool is for.
    """

    __tablename__ = "holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "ticker", name="uq_holding_portfolio_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ticker: Mapped[str] = mapped_column(
        ForeignKey("companies.ticker", ondelete="RESTRICT"), nullable=False, index=True
    )
    quantity_milli: Mapped[int] = mapped_column(Integer, nullable=False)
    average_cost_paise: Mapped[int] = mapped_column(Integer, nullable=False)
    first_bought_on: Mapped[date | None] = mapped_column(Date, nullable=True)

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")
    company: Mapped[Company] = relationship(back_populates="holdings")

    # --- Friendly views over the integer storage ---------------------------

    @property
    def quantity(self) -> Decimal:
        """Shares held, fractional units included."""
        return from_milli(self.quantity_milli)

    @property
    def average_cost(self) -> Decimal:
        """Weighted average purchase price per share, in rupees."""
        return from_paise(self.average_cost_paise)

    @property
    def invested_paise(self) -> int:
        """Total cost basis in paise, computed in exact integer arithmetic."""
        total = (Decimal(self.quantity_milli) * Decimal(self.average_cost_paise)) / MILLI_PER_UNIT
        return int(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    @property
    def invested(self) -> Decimal:
        """Total cost basis in rupees.

        Cost basis, not market value: this project has no price feed, and
        reporting a cost as though it were a valuation would be a lie the user
        could act on.
        """
        return from_paise(self.invested_paise)

    @property
    def sector(self) -> str:
        return self.company.sector

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Holding(ticker={self.ticker!r}, quantity={self.quantity}, invested={self.invested})"


class Alert(Base):
    """A development the sources show bears on one holding.

    Rows are written only for alerts that were **grounded** — the model named a
    development and cited at least one supplied source. An ungrounded judgement
    produces no row, which turns the defect recorded in CLAUDE.md section 11
    (no-impact answers cite nothing) into a safe failure: silence rather than an
    unverifiable warning.

    ``fingerprint`` is a hash of the holding plus the evidence cited, so the same
    circular seen on ten consecutive runs raises one alert, not ten. New evidence
    about the same holding is a new fingerprint and a new alert, which is what
    makes a genuinely new development visible.
    """

    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("portfolio_id", "fingerprint", name="uq_alert_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The holding this concerns. Nullable so a portfolio-wide alert is possible
    #: later without a migration.
    ticker: Mapped[str | None] = mapped_column(
        ForeignKey("companies.ticker", ondelete="RESTRICT"), nullable=True, index=True
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    #: Citations as JSON, so an alert can be audited long after the indexes move on.
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    portfolio: Mapped[Portfolio] = relationship()

    @property
    def citations(self) -> list[dict[str, Any]]:
        """The evidence, decoded. Never raises on a malformed row."""
        try:
            data = json.loads(self.evidence)
        except (TypeError, ValueError):
            return []
        return data if isinstance(data, list) else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "summary": self.summary,
            "citations": self.citations,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "acknowledged": self.acknowledged,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Alert(ticker={self.ticker!r}, summary={self.summary[:40]!r})"
