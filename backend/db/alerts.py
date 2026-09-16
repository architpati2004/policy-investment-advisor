"""Reading and acknowledging stored alerts.

Kept beside the portfolio operations for the same reason they were: the engine
decides *what* is an alert, this module only stores and retrieves them, and API
routes get a function rather than a query.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Alert
from backend.db.portfolio import DEFAULT_PORTFOLIO, get_portfolio
from backend.exceptions import PortfolioError
from backend.logging_config import get_logger

logger = get_logger(__name__)


def list_alerts(
    session: Session,
    *,
    portfolio_name: str = DEFAULT_PORTFOLIO,
    ticker: str | None = None,
    include_acknowledged: bool = False,
    limit: int | None = None,
) -> list[Alert]:
    """Alerts for a portfolio, newest first."""
    portfolio = get_portfolio(session, portfolio_name)
    query = select(Alert).where(Alert.portfolio_id == portfolio.id)
    if ticker:
        query = query.where(Alert.ticker == ticker.strip().upper())
    if not include_acknowledged:
        query = query.where(Alert.acknowledged.is_(False))
    query = query.order_by(Alert.created_at.desc(), Alert.id.desc())
    if limit:
        query = query.limit(limit)
    return list(session.scalars(query))


def acknowledge(
    session: Session,
    alert_id: int,
    *,
    portfolio_name: str = DEFAULT_PORTFOLIO,
) -> Alert:
    """Mark one alert as seen.

    Acknowledged rather than deleted: the fingerprint has to stay, or the next
    run would raise the same alert again on the same evidence.

    Raises:
        PortfolioError: no such alert in this portfolio.
    """
    portfolio = get_portfolio(session, portfolio_name)
    alert = session.scalar(
        select(Alert).where(Alert.id == alert_id, Alert.portfolio_id == portfolio.id)
    )
    if alert is None:
        raise PortfolioError(
            f"No alert {alert_id} in {portfolio_name!r}",
            remediation="List what there is: python scripts/alerts.py list",
        )
    alert.acknowledged = True
    session.flush()
    logger.info("Acknowledged alert %d", alert_id)
    return alert


def acknowledge_all(session: Session, *, portfolio_name: str = DEFAULT_PORTFOLIO) -> int:
    """Mark every outstanding alert as seen. Returns how many."""
    outstanding = list_alerts(session, portfolio_name=portfolio_name)
    for alert in outstanding:
        alert.acknowledged = True
    session.flush()
    return len(outstanding)


def clear_alerts(session: Session, *, portfolio_name: str = DEFAULT_PORTFOLIO) -> int:
    """Delete every alert for a portfolio, fingerprints included.

    Deliberately separate from acknowledging: this forgets the evidence too, so
    the next run can raise the same alerts again. Useful when re-testing, and
    wrong as a way of dismissing something.
    """
    portfolio = get_portfolio(session, portfolio_name)
    alerts = list(session.scalars(select(Alert).where(Alert.portfolio_id == portfolio.id)))
    for alert in alerts:
        session.delete(alert)
    session.flush()
    logger.info("Cleared %d alerts from %r", len(alerts), portfolio_name)
    return len(alerts)
