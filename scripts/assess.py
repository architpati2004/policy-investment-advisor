"""Phase 8 tool: ask what a policy change means for the portfolio.

Pulls policy and company context together and reports which holdings a change
bears on — and, just as importantly, which it does not.

Run with::

    python scripts/assess.py "how do the new RBI deposit rate rules affect my holdings?"
    python scripts/assess.py "what regulatory risks affect consumer goods?" --show-context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.db import portfolio as portfolio_service  # noqa: E402
from backend.db.session import session_scope  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.rag.portfolio_rag import PortfolioAssessment, build_portfolio_rag  # noqa: E402

PASS, FAIL, INFO, WARN = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]"
PREVIEW_CHARS = 200


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def _print(assessment: PortfolioAssessment, show_context: bool) -> None:
    sys.stdout.write("\n" + "=" * 74 + "\n")
    if assessment.covered:
        # Already prose: the chain strips its own sentinels after parsing them.
        sys.stdout.write(assessment.answer.strip() + "\n")
    else:
        sys.stdout.write(f"NOT COVERED BY THE INDEXED SOURCES\n  {assessment.reason}\n")
    sys.stdout.write("=" * 74 + "\n\n")

    if assessment.covered:
        if not assessment.impact_declared:
            # Silence is not "nothing affected": say the model did not answer it.
            _report(WARN, "the assessment did not declare which holdings are affected")
        elif assessment.affected:
            _report(PASS, f"affected holdings: {', '.join(assessment.affected)}")
            if assessment.unaffected:
                _report(INFO, f"unaffected: {', '.join(assessment.unaffected)}")
        else:
            _report(INFO, "no holding is affected according to the sources")

    stage = "generation" if assessment.generated else "retrieval, no model call"
    verb = "assessed" if assessment.covered else "declined"
    _report(INFO, f"{verb} in {assessment.elapsed_seconds:.1f}s at {stage}")
    _report(
        INFO,
        f"context: {assessment.policy_chunks} policy + {assessment.company_chunks} company chunks",
    )

    if assessment.invalid_citations:
        _report(WARN, f"cited sources that do not exist: {assessment.invalid_citations}")
    if assessment.covered and not assessment.cited_sources:
        _report(WARN, "the assessment cites no source — treat it as unverified")

    for source in assessment.sources:
        mark = "cited" if source.cited else "     "
        origin = str(source.metadata.get("origin", "?")).upper()
        sys.stdout.write(f"  [{source.number}] {mark} {origin:<8} {source.score:.3f}  {source.citation}\n")
        if show_context:
            preview = " ".join(source.text.split())[:PREVIEW_CHARS]
            sys.stdout.write(f"        {preview}...\n")

    sys.stdout.write("\n")
    _report(INFO, "Research aid grounded in the indexed documents — not investment advice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess a question against the portfolio.")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument(
        "--portfolio", default=portfolio_service.DEFAULT_PORTFOLIO, help="Portfolio name"
    )
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    sys.stdout.write("Phase 8 portfolio impact\n" + "-" * 74 + "\n")
    _report(INFO, f'Question: "{args.question}"')
    _report(INFO, f"model={settings.llm_model} | portfolio={args.portfolio}")
    _report(INFO, "Generating locally; the first call loads the model into RAM...")

    try:
        with session_scope(settings) as session:
            assessment = build_portfolio_rag(settings).assess(
                args.question, session, portfolio_name=args.portfolio
            )
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    _print(assessment, args.show_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
