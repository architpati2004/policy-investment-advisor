"""Portfolio operations.

Everything the rest of the application does to a portfolio lives here, taking a
``Session`` rather than opening one, so the CLI, the Phase 11 API and the tests
each control their own transaction. No SQL and no model imports leak into route
handlers.

The function that matters beyond this phase is :func:`portfolio_scope`: it turns
a portfolio into the tickers and sectors it covers, which is exactly the shape
:func:`backend.rag.retrieval.scope_filter` takes. That is the join between
"what I own" and "which documents are about it", and it is why holdings are
keyed on the same ticker the company index writes into chunk metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.db.models import Company, Holding, Portfolio, from_paise, to_milli, to_paise
from backend.exceptions import PortfolioError, UnknownTickerError
from backend.ingestion.company_registry import CompanyRegistry
from backend.logging_config import get_logger

logger = get_logger(__name__)

#: Used when no portfolio is named. One portfolio is the common case; the schema
#: still supports more, so adding a second never needs a migration.
DEFAULT_PORTFOLIO = "Demo Portfolio"


@dataclass(frozen=True)
class PortfolioScope:
    """The companies and sectors a portfolio covers.

    Passed straight to ``scope_filter(companies=..., sectors=...)`` in Phase 8.
    Sectors matter independently of companies: a rule aimed at an industry
    reaches a holder whose company is never named in it.
    """

    tickers: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.tickers and not self.sectors

    def to_dict(self) -> dict[str, list[str]]:
        return {"tickers": list(self.tickers), "sectors": list(self.sectors)}


@dataclass(frozen=True)
class HoldingSummary:
    """One holding, with its share of the portfolio by cost."""

    ticker: str
    name: str
    sector: str
    quantity: Decimal
    average_cost: Decimal
    invested: Decimal
    weight: Decimal
    first_bought_on: date | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "quantity": str(self.quantity),
            "average_cost": str(self.average_cost),
            "invested": str(self.invested),
            "weight_percent": str(self.weight),
            "first_bought_on": self.first_bought_on.isoformat() if self.first_bought_on else None,
        }


@dataclass(frozen=True)
class PortfolioSummary:
    """A portfolio's holdings and how its cost is distributed.

    Everything here is **cost basis**, never market value: the project has no
    price feed, and presenting cost as a valuation would be a number a user
    could act on and be wrong about.
    """

    name: str
    holdings: list[HoldingSummary] = field(default_factory=list)
    invested: Decimal = Decimal("0.00")

    @property
    def sector_weights(self) -> dict[str, Decimal]:
        weights: dict[str, Decimal] = {}
        for holding in self.holdings:
            weights[holding.sector] = weights.get(holding.sector, Decimal("0")) + holding.weight
        return dict(sorted(weights.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "invested": str(self.invested),
            "holdings": [holding.to_dict() for holding in self.holdings],
            "sector_weights": {k: str(v) for k, v in self.sector_weights.items()},
        }


# --- Companies ---------------------------------------------------------------


def sync_companies(session: Session, registry: CompanyRegistry) -> tuple[int, int]:
    """Mirror the registry into the database.

    The registry file stays the source of truth for who a company is; this copies
    it in so holdings have something to reference and so sector queries are SQL.

    Returns:
        ``(added, updated)`` counts.
    """
    added = updated = 0
    for declared in registry:
        existing = session.get(Company, declared.ticker)
        if existing is None:
            session.add(Company(ticker=declared.ticker, name=declared.name, sector=declared.sector))
            added += 1
        elif (existing.name, existing.sector) != (declared.name, declared.sector):
            existing.name, existing.sector = declared.name, declared.sector
            updated += 1

    session.flush()
    logger.info("Synced companies from registry: %d added, %d updated", added, updated)
    return added, updated


def known_tickers(session: Session) -> list[str]:
    return list(session.scalars(select(Company.ticker).order_by(Company.ticker)))


# --- Portfolios --------------------------------------------------------------


def get_or_create_portfolio(session: Session, name: str = DEFAULT_PORTFOLIO) -> Portfolio:
    """Fetch a portfolio by name, creating it if it does not exist."""
    portfolio = session.scalar(select(Portfolio).where(Portfolio.name == name))
    if portfolio is None:
        portfolio = Portfolio(name=name)
        session.add(portfolio)
        session.flush()
        logger.info("Created portfolio %r", name)
    return portfolio


def get_portfolio(session: Session, name: str = DEFAULT_PORTFOLIO) -> Portfolio:
    """Fetch a portfolio with its holdings and companies loaded.

    Raises:
        PortfolioError: no portfolio of that name exists.
    """
    portfolio = session.scalar(
        select(Portfolio)
        .where(Portfolio.name == name)
        .options(selectinload(Portfolio.holdings).selectinload(Holding.company))
    )
    if portfolio is None:
        raise PortfolioError(
            f"No portfolio named {name!r}",
            remediation="Create one: python scripts/portfolio.py add --ticker ... --quantity ...",
        )
    return portfolio


# --- Holdings ----------------------------------------------------------------


def add_holding(
    session: Session,
    ticker: str,
    quantity: Decimal | float | int | str,
    average_cost: Decimal | float | int | str,
    *,
    portfolio_name: str = DEFAULT_PORTFOLIO,
    bought_on: date | None = None,
) -> Holding:
    """Add a position, or add to one already held.

    Buying more of something held already re-weights the average cost rather
    than creating a second row:

        (old_qty x old_cost + new_qty x new_cost) / (old_qty + new_qty)

    computed in integer paise so repeated purchases cannot drift.

    Raises:
        UnknownTickerError: the ticker is not a known company.
        PortfolioError: quantity or cost is not positive.
    """
    ticker = ticker.strip().upper()
    company = session.get(Company, ticker)
    if company is None:
        raise UnknownTickerError(ticker, known_tickers(session))

    quantity_milli = to_milli(quantity)
    cost_paise = to_paise(average_cost)
    if quantity_milli <= 0:
        raise PortfolioError(
            f"Quantity must be positive, got {quantity}",
            remediation="To reduce or close a position, use 'remove' instead.",
        )
    if cost_paise <= 0:
        raise PortfolioError(
            f"Average cost must be positive, got {average_cost}",
            remediation="Pass the price paid per share, in rupees.",
        )

    portfolio = get_or_create_portfolio(session, portfolio_name)
    existing = session.scalar(
        select(Holding).where(Holding.portfolio_id == portfolio.id, Holding.ticker == ticker)
    )

    if existing is None:
        holding = Holding(
            ticker=ticker,
            quantity_milli=quantity_milli,
            average_cost_paise=cost_paise,
            first_bought_on=bought_on,
        )
        # Appended to the relationship rather than added to the session, so an
        # already-loaded ``portfolio.holdings`` sees the new position too. A bare
        # ``session.add`` writes the row but leaves that collection stale, and
        # the next read in the same transaction would be missing it.
        portfolio.holdings.append(holding)
        session.flush()
        logger.info("Added %s x%s to %r", ticker, holding.quantity, portfolio_name)
        return holding

    total_milli = existing.quantity_milli + quantity_milli
    blended = (
        existing.quantity_milli * existing.average_cost_paise + quantity_milli * cost_paise
    ) // total_milli
    existing.quantity_milli = total_milli
    existing.average_cost_paise = blended
    if bought_on and (existing.first_bought_on is None or bought_on < existing.first_bought_on):
        existing.first_bought_on = bought_on
    session.flush()
    logger.info("Increased %s to x%s at average %s", ticker, existing.quantity, existing.average_cost)
    return existing


def remove_holding(
    session: Session,
    ticker: str,
    *,
    portfolio_name: str = DEFAULT_PORTFOLIO,
) -> None:
    """Close a position entirely.

    Raises:
        PortfolioError: the portfolio does not hold that ticker.
    """
    ticker = ticker.strip().upper()
    portfolio = get_portfolio(session, portfolio_name)
    holding = session.scalar(
        select(Holding).where(Holding.portfolio_id == portfolio.id, Holding.ticker == ticker)
    )
    if holding is None:
        raise PortfolioError(
            f"{portfolio_name!r} does not hold {ticker}",
            remediation="List what it does hold: python scripts/portfolio.py list",
        )
    # Removed from the relationship, not just deleted: the delete-orphan cascade
    # issues the DELETE, and the portfolio's loaded collection stops reporting a
    # position that is gone.
    portfolio.holdings.remove(holding)
    session.flush()
    logger.info("Removed %s from %r", ticker, portfolio_name)


def list_holdings(session: Session, portfolio_name: str = DEFAULT_PORTFOLIO) -> list[Holding]:
    """Holdings in ticker order."""
    return list(get_portfolio(session, portfolio_name).holdings)


# --- Derived views -----------------------------------------------------------


def portfolio_scope(session: Session, portfolio_name: str = DEFAULT_PORTFOLIO) -> PortfolioScope:
    """The tickers and sectors a portfolio covers.

    The hand-off to retrieval: pass these to ``scope_filter`` to restrict either
    index to documents about what is actually held.
    """
    holdings = list_holdings(session, portfolio_name)
    tickers = tuple(sorted({holding.ticker for holding in holdings}))
    sectors = tuple(sorted({holding.company.sector for holding in holdings}))
    return PortfolioScope(tickers=tickers, sectors=sectors)


def summarise(session: Session, portfolio_name: str = DEFAULT_PORTFOLIO) -> PortfolioSummary:
    """Holdings with cost-basis weights, largest position first."""
    portfolio = get_portfolio(session, portfolio_name)
    total_paise = sum(holding.invested_paise for holding in portfolio.holdings)

    summaries: list[HoldingSummary] = []
    for holding in portfolio.holdings:
        weight = (
            (Decimal(holding.invested_paise) * 100 / Decimal(total_paise)).quantize(Decimal("0.01"))
            if total_paise
            else Decimal("0.00")
        )
        summaries.append(
            HoldingSummary(
                ticker=holding.ticker,
                name=holding.company.name,
                sector=holding.company.sector,
                quantity=holding.quantity,
                average_cost=holding.average_cost,
                invested=holding.invested,
                weight=weight,
                first_bought_on=holding.first_bought_on,
            )
        )

    summaries.sort(key=lambda item: item.invested, reverse=True)
    return PortfolioSummary(
        name=portfolio.name, holdings=summaries, invested=from_paise(total_paise)
    )


# --- Demo data ---------------------------------------------------------------

#: Illustrative holding, not a real position. GODREJCP is the only company the
#: corpus currently covers; the shape is identical for any other ticker, which
#: is the point — extending this list needs no schema or code change.
DEMO_HOLDINGS: Sequence[tuple[str, str, str]] = (("GODREJCP", "150", "1180.50"),)


def seed_demo(
    session: Session,
    registry: CompanyRegistry,
    *,
    portfolio_name: str = DEFAULT_PORTFOLIO,
    holdings: Iterable[tuple[str, str, str]] = DEMO_HOLDINGS,
) -> PortfolioSummary:
    """Create the demo portfolio, skipping any ticker the registry lacks.

    Skipping rather than failing is what makes this list extensible: adding
    ``("HDFCBANK", "40", "1650.00")`` here works the moment HDFC Bank is
    declared in the registry, and is ignored harmlessly until then.
    """
    sync_companies(session, registry)
    for ticker, quantity, cost in holdings:
        if session.get(Company, ticker) is None:
            logger.warning("Skipping demo holding %s: not in the company registry", ticker)
            continue
        add_holding(session, ticker, quantity, cost, portfolio_name=portfolio_name)
    return summarise(session, portfolio_name)
