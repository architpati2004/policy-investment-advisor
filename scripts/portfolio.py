"""Phase 7 tool: manage the portfolio database.

The portfolio is what turns a document search into advice: it decides which
companies and sectors a policy change is checked against. This CLI is how it
gets populated until the Phase 12 frontend exists.

Run with::

    python scripts/portfolio.py init            # create tables, import the registry
    python scripts/portfolio.py seed-demo       # illustrative GODREJCP position
    python scripts/portfolio.py add --ticker GODREJCP --quantity 150 --cost 1180.50
    python scripts/portfolio.py list
    python scripts/portfolio.py scope           # what Phase 8 will retrieve against
    python scripts/portfolio.py remove --ticker GODREJCP
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Settings, get_settings  # noqa: E402
from backend.db import portfolio as service  # noqa: E402
from backend.db.session import init_db, session_scope  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.ingestion.company_registry import CompanyRegistry  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

PASS, FAIL, INFO, WARN = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]"


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def _print_summary(summary: service.PortfolioSummary) -> None:
    if not summary.holdings:
        _report(WARN, f"{summary.name!r} holds nothing yet")
        return

    sys.stdout.write(f"\n{summary.name}\n" + "-" * 78 + "\n")
    sys.stdout.write(f"{'TICKER':<12}{'SECTOR':<16}{'QTY':>10}{'AVG COST':>12}{'INVESTED':>14}{'WEIGHT':>9}\n")
    for holding in summary.holdings:
        sys.stdout.write(
            f"{holding.ticker:<12}{holding.sector:<16}{holding.quantity:>10}"
            f"{holding.average_cost:>12}{holding.invested:>14}{holding.weight:>8}%\n"
        )
    sys.stdout.write("-" * 78 + "\n")
    sys.stdout.write(f"{'TOTAL COST':<38}{summary.invested:>26}\n")
    weights = ", ".join(f"{sector} {weight}%" for sector, weight in summary.sector_weights.items())
    sys.stdout.write(f"by sector: {weights}\n\n")
    _report(INFO, "Cost basis only — this project has no price feed, so nothing here is a valuation")


def init(args: argparse.Namespace, settings: Settings) -> int:
    """Create the schema and import the company registry."""
    init_db(settings=settings)
    registry = CompanyRegistry.load(settings=settings)
    with session_scope(settings) as session:
        added, updated = service.sync_companies(session, registry)
    _report(PASS, f"Database ready at {settings.resolved_database_url}")
    _report(PASS, f"Companies: {added} added, {updated} updated ({len(registry)} in the registry)")
    return 0


def sync(args: argparse.Namespace, settings: Settings) -> int:
    """Re-import the registry after editing it."""
    registry = CompanyRegistry.load(settings=settings)
    with session_scope(settings) as session:
        added, updated = service.sync_companies(session, registry)
        known = service.known_tickers(session)
    _report(PASS, f"{added} added, {updated} updated; known tickers: {', '.join(known) or 'none'}")
    return 0


def seed_demo(args: argparse.Namespace, settings: Settings) -> int:
    """Create the illustrative demo portfolio."""
    init_db(settings=settings)
    registry = CompanyRegistry.load(settings=settings)
    with session_scope(settings) as session:
        summary = service.seed_demo(session, registry, portfolio_name=args.portfolio)
    _print_summary(summary)
    _report(INFO, "Illustrative holdings, not real positions")
    return 0


def add(args: argparse.Namespace, settings: Settings) -> int:
    """Add a position, or add to one already held."""
    init_db(settings=settings)
    bought_on = date.fromisoformat(args.bought_on) if args.bought_on else None
    with session_scope(settings) as session:
        holding = service.add_holding(
            session,
            args.ticker,
            args.quantity,
            args.cost,
            portfolio_name=args.portfolio,
            bought_on=bought_on,
        )
        _report(PASS, f"{holding.ticker}: {holding.quantity} at average {holding.average_cost}")
        _print_summary(service.summarise(session, args.portfolio))
    return 0


def remove(args: argparse.Namespace, settings: Settings) -> int:
    """Close a position."""
    with session_scope(settings) as session:
        service.remove_holding(session, args.ticker, portfolio_name=args.portfolio)
        _report(PASS, f"Removed {args.ticker.upper()}")
        _print_summary(service.summarise(session, args.portfolio))
    return 0


def show(args: argparse.Namespace, settings: Settings) -> int:
    """Print the holdings and their weights."""
    with session_scope(settings) as session:
        _print_summary(service.summarise(session, args.portfolio))
    return 0


def scope(args: argparse.Namespace, settings: Settings) -> int:
    """Print what Phase 8 will restrict retrieval to."""
    with session_scope(settings) as session:
        derived = service.portfolio_scope(session, args.portfolio)
    if derived.is_empty:
        _report(WARN, f"{args.portfolio!r} holds nothing, so retrieval would not be restricted")
        return 0
    _report(PASS, f"companies: {', '.join(derived.tickers)}")
    _report(PASS, f"sectors:   {', '.join(derived.sectors)}")
    _report(INFO, "Phase 8 passes these to scope_filter() to match documents against holdings")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the portfolio database.")
    parser.add_argument(
        "--portfolio",
        default=service.DEFAULT_PORTFOLIO,
        help=f"Portfolio name (default: {service.DEFAULT_PORTFOLIO!r})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create tables and import the registry").set_defaults(handler=init)
    sub.add_parser("sync", help="Re-import the company registry").set_defaults(handler=sync)
    sub.add_parser("seed-demo", help="Create the illustrative demo portfolio").set_defaults(
        handler=seed_demo
    )
    sub.add_parser("list", help="Show holdings and weights").set_defaults(handler=show)
    sub.add_parser("scope", help="Show the companies and sectors held").set_defaults(handler=scope)

    adder = sub.add_parser("add", help="Add or increase a position")
    adder.add_argument("--ticker", required=True)
    adder.add_argument("--quantity", required=True, help="Shares, fractional allowed")
    adder.add_argument("--cost", required=True, help="Price paid per share, in rupees")
    adder.add_argument("--bought-on", default=None, help="Purchase date (YYYY-MM-DD)")
    adder.set_defaults(handler=add)

    remover = sub.add_parser("remove", help="Close a position")
    remover.add_argument("--ticker", required=True)
    remover.set_defaults(handler=remove)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    sys.stdout.write(f"Phase 7 portfolio — {args.command}\n" + "-" * 78 + "\n")
    try:
        return int(args.handler(args, settings))
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    except ValueError as exc:
        _report(FAIL, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
