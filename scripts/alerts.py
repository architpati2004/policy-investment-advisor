"""Phase 10 tool: check the portfolio for developments worth knowing about.

Each run checks every holding against policy, company filings and news, and
records an alert only where the model names a development *and* cites a source
for it. Most runs on a small corpus will report nothing, which is the intended
outcome — an alert engine that fires constantly is noise.

Run with::

    python scripts/alerts.py run
    python scripts/alerts.py run --ticker GODREJCP
    python scripts/alerts.py list
    python scripts/alerts.py ack --id 3
    python scripts/alerts.py clear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.alerts.engine import RunReport, build_engine  # noqa: E402
from backend.config import Settings, get_settings  # noqa: E402
from backend.db import alerts as alert_store  # noqa: E402
from backend.db import portfolio as portfolio_service  # noqa: E402
from backend.db.session import init_db, session_scope  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

PASS, FAIL, INFO, WARN, ALERT = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]", "[ALERT]"


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def _print_run(report: RunReport) -> None:
    sys.stdout.write("\n")
    for finding in report.raised:
        sys.stdout.write("=" * 74 + "\n")
        sys.stdout.write(f"{ALERT} {finding.ticker}\n{finding.summary}\n")
        for citation in finding.citations:
            origin = str(citation.get("origin", "?")).upper()
            sys.stdout.write(f"    [{origin}] {citation['citation']}\n")
            if citation.get("url"):
                sys.stdout.write(f"          {citation['url']}\n")
        sys.stdout.write("=" * 74 + "\n\n")

    for finding in report.quiet:
        note = finding.reason or "nothing material"
        marker = WARN if finding.duplicate is False and finding.summary else INFO
        sys.stdout.write(f"{marker} {finding.ticker}: {note}")
        sys.stdout.write(f" ({finding.considered} chunks, {finding.elapsed_seconds:.1f}s)\n")

    sys.stdout.write("\n")
    _report(
        PASS if report.raised else INFO,
        f"{len(report.raised)} alert(s) from {len(report.findings)} holding(s) "
        f"in {report.elapsed_seconds:.1f}s",
    )
    if not report.raised:
        _report(INFO, "Nothing raised — on a small corpus that is the expected result")
    _report(INFO, "Grounded in the indexed documents — not investment advice")


def run(args: argparse.Namespace, settings: Settings) -> int:
    """Check every holding (or one) and record what is material."""
    init_db(settings=settings)
    engine = build_engine(settings)

    _report(INFO, f"model={settings.llm_model} | portfolio={args.portfolio}")
    _report(INFO, "One local generation per holding; the first loads the model into RAM...")

    with session_scope(settings) as session:
        summary = portfolio_service.summarise(session, args.portfolio)
        holdings = [
            holding
            for holding in summary.holdings
            if not args.ticker or holding.ticker == args.ticker.strip().upper()
        ]
        if not holdings:
            _report(FAIL, f"No holdings to check in {args.portfolio!r}")
            return 1

        portfolio = portfolio_service.get_portfolio(session, args.portfolio)
        findings = [engine.check(holding, session, portfolio.id) for holding in holdings]
        report = RunReport(
            portfolio=args.portfolio,
            findings=findings,
            elapsed_seconds=sum(f.elapsed_seconds for f in findings),
        )
        _print_run(report)
    return 0


def show(args: argparse.Namespace, settings: Settings) -> int:
    """List outstanding alerts."""
    with session_scope(settings) as session:
        stored = alert_store.list_alerts(
            session,
            portfolio_name=args.portfolio,
            ticker=args.ticker,
            include_acknowledged=args.all,
        )
        if not stored:
            _report(INFO, "No outstanding alerts")
            return 0
        for alert in stored:
            seen = " (acknowledged)" if alert.acknowledged else ""
            sys.stdout.write(f"\n#{alert.id} {alert.ticker}{seen}\n  {alert.summary}\n")
            for citation in alert.citations:
                sys.stdout.write(f"    [{str(citation.get('origin', '?')).upper()}] "
                                 f"{citation.get('citation')}\n")
        sys.stdout.write("\n")
        _report(PASS, f"{len(stored)} alert(s)")
    return 0


def ack(args: argparse.Namespace, settings: Settings) -> int:
    """Mark alerts as seen, keeping their fingerprints."""
    with session_scope(settings) as session:
        if args.id:
            alert = alert_store.acknowledge(session, args.id, portfolio_name=args.portfolio)
            _report(PASS, f"Acknowledged #{alert.id} ({alert.ticker})")
        else:
            count = alert_store.acknowledge_all(session, portfolio_name=args.portfolio)
            _report(PASS, f"Acknowledged {count} alert(s)")
    return 0


def clear(args: argparse.Namespace, settings: Settings) -> int:
    """Delete alerts and their fingerprints, so they can be raised again."""
    with session_scope(settings) as session:
        count = alert_store.clear_alerts(session, portfolio_name=args.portfolio)
    _report(PASS, f"Deleted {count} alert(s) — the same evidence can raise them again")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the portfolio for material developments.")
    parser.add_argument("--portfolio", default=portfolio_service.DEFAULT_PORTFOLIO)
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="Check holdings and record alerts")
    runner.add_argument("--ticker", default=None, help="Check only this holding")
    runner.set_defaults(handler=run)

    lister = sub.add_parser("list", help="Show outstanding alerts")
    lister.add_argument("--ticker", default=None)
    lister.add_argument("--all", action="store_true", help="Include acknowledged alerts")
    lister.set_defaults(handler=show)

    acker = sub.add_parser("ack", help="Mark alerts as seen")
    acker.add_argument("--id", type=int, default=None, help="One alert; omit for all")
    acker.set_defaults(handler=ack)

    sub.add_parser("clear", help="Delete alerts and their fingerprints").set_defaults(handler=clear)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    sys.stdout.write(f"Phase 10 alerts — {args.command}\n" + "-" * 74 + "\n")
    try:
        return int(args.handler(args, settings))
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
