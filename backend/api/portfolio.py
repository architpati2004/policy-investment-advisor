"""Portfolio endpoints.

Thin: every operation is a call into :mod:`backend.db.portfolio`, which already
owns the rules — weighted average cost on a repeat purchase, refusing an unknown
ticker, deriving the retrieval scope. Routes translate HTTP to that and back,
and hold no logic of their own.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.api import schemas
from backend.api.deps import get_db
from backend.db import portfolio as service
from backend.ingestion.company_registry import CompanyRegistry
from backend.config import Settings, get_settings

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("", response_model=schemas.PortfolioOut)
def read_portfolio(
    name: str = service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.PortfolioOut:
    """Holdings and cost-basis weights.

    Raises 404 through the application error handler when the portfolio does not
    exist, rather than returning an empty portfolio that looks like a real one.
    """
    return schemas.PortfolioOut(**service.summarise(session, name).to_dict())


@router.post("/holdings", response_model=schemas.PortfolioOut, status_code=status.HTTP_201_CREATED)
def add_holding(
    holding: schemas.HoldingIn,
    name: str = service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.PortfolioOut:
    """Add a position, or add to one already held.

    Buying more of something held re-weights the average cost rather than
    creating a second row; the response is the whole portfolio, because the
    weights of every other holding have just changed too.
    """
    service.add_holding(
        session,
        holding.ticker,
        holding.quantity,
        holding.average_cost,
        portfolio_name=name,
        bought_on=date.fromisoformat(holding.bought_on) if holding.bought_on else None,
    )
    return schemas.PortfolioOut(**service.summarise(session, name).to_dict())


@router.delete("/holdings/{ticker}", response_model=schemas.PortfolioOut)
def remove_holding(
    ticker: str,
    name: str = service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.PortfolioOut:
    """Close a position."""
    service.remove_holding(session, ticker, portfolio_name=name)
    return schemas.PortfolioOut(**service.summarise(session, name).to_dict())


@router.get("/scope", response_model=schemas.ScopeOut)
def read_scope(
    name: str = service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.ScopeOut:
    """The companies and sectors retrieval is restricted to for this portfolio."""
    return schemas.ScopeOut(**service.portfolio_scope(session, name).to_dict())


@router.get("/companies", response_model=list[schemas.CompanyOut])
def read_companies(settings: Settings = Depends(get_settings)) -> list[schemas.CompanyOut]:
    """Companies that may be held, from the registry.

    Served from the registry file rather than the database mirror, because the
    file is the source of truth and a company declared but not yet synced should
    still be offerable to a user.
    """
    registry = CompanyRegistry.load(settings=settings)
    return [
        schemas.CompanyOut(ticker=c.ticker, name=c.name, sector=c.sector) for c in registry
    ]
