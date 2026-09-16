"""Alert endpoints.

``POST /run`` is the slow one and deliberately synchronous: it costs one local
generation *per holding*, so a six-holding portfolio is minutes, not seconds.
A background-job queue would hide that, and hiding it is how a user ends up
believing a stale alert list is current. The ``ticker`` parameter exists so a
client can check one holding without paying for all of them.

A run returns its quiet holdings as well as its alerts. "We checked and found
nothing" and "we did not check" are different statements, and a response
carrying only raised alerts cannot distinguish them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.alerts.engine import AlertEngine, RunReport
from backend.api import schemas
from backend.api.deps import get_alert_engine, get_db
from backend.db import alerts as alert_store
from backend.db import portfolio as portfolio_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[schemas.AlertOut])
def read_alerts(
    portfolio: str = portfolio_service.DEFAULT_PORTFOLIO,
    ticker: str | None = None,
    include_acknowledged: bool = False,
    limit: int | None = Query(None, ge=1, le=200),
    session: Session = Depends(get_db),
) -> list[schemas.AlertOut]:
    """Stored alerts, newest first."""
    stored = alert_store.list_alerts(
        session,
        portfolio_name=portfolio,
        ticker=ticker,
        include_acknowledged=include_acknowledged,
        limit=limit,
    )
    return [schemas.AlertOut(**alert.to_dict()) for alert in stored]


@router.post("/run", response_model=schemas.AlertRunOut)
def run_alerts(
    portfolio: str = portfolio_service.DEFAULT_PORTFOLIO,
    ticker: str | None = None,
    engine: AlertEngine = Depends(get_alert_engine),
    session: Session = Depends(get_db),
) -> schemas.AlertRunOut:
    """Check holdings for material developments.

    One local generation per holding checked. Expect tens of seconds each.
    """
    summary = portfolio_service.summarise(session, portfolio)
    record = portfolio_service.get_portfolio(session, portfolio)
    wanted = ticker.strip().upper() if ticker else None

    findings = [
        engine.check(holding, session, record.id)
        for holding in summary.holdings
        if wanted is None or holding.ticker == wanted
    ]
    report = RunReport(
        portfolio=portfolio,
        findings=findings,
        elapsed_seconds=sum(finding.elapsed_seconds for finding in findings),
    )
    return schemas.AlertRunOut(**report.to_dict())


@router.post("/{alert_id}/acknowledge", response_model=schemas.AlertOut)
def acknowledge_alert(
    alert_id: int,
    portfolio: str = portfolio_service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.AlertOut:
    """Mark one alert as seen, keeping its fingerprint so it cannot return."""
    return schemas.AlertOut(**alert_store.acknowledge(session, alert_id, portfolio_name=portfolio).to_dict())


@router.delete("", response_model=schemas.MessageOut)
def clear_alerts(
    portfolio: str = portfolio_service.DEFAULT_PORTFOLIO,
    session: Session = Depends(get_db),
) -> schemas.MessageOut:
    """Delete alerts *and their fingerprints*, so the same evidence can raise them again.

    Distinct from acknowledging, which keeps the fingerprint. This is for
    re-testing, not for dismissing something.
    """
    count = alert_store.clear_alerts(session, portfolio_name=portfolio)
    return schemas.MessageOut(message=f"Deleted {count} alert(s); the same evidence can raise them again")
