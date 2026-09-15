"""Phase 5 tool: ask the policy corpus a question and see what grounds the answer.

Prints the answer, the sources it cites with their scores, and — when the corpus
cannot answer — why it declined and at which stage. The refusal path is the one
worth watching: a system that answers everything is not grounded, it is guessing.

Run with::

    python scripts/ask_policy.py "can a bank pay interest on a current account?"
    python scripts/ask_policy.py "what are SEBI's mutual fund rules?" --show-context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402
from backend.rag.policy_rag import PolicyAnswer, build_policy_rag  # noqa: E402

PASS, FAIL, INFO, WARN = "[ OK ]", "[FAIL]", "[INFO]", "[WARN]"
PREVIEW_CHARS = 240


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def _print_answer(answer: PolicyAnswer, show_context: bool) -> None:
    sys.stdout.write("\n" + "=" * 70 + "\n")
    if answer.covered:
        sys.stdout.write(answer.answer.strip() + "\n")
    else:
        sys.stdout.write(f"NOT COVERED BY THE INDEXED SOURCES\n  {answer.reason}\n")
    sys.stdout.write("=" * 70 + "\n\n")

    timing = f"{answer.elapsed_seconds:.1f}s"
    if answer.covered:
        _report(INFO, f"answered in {timing}")
    else:
        stage = "generation" if answer.generated else "retrieval, no model call"
        _report(INFO, f"refused in {timing} at {stage}")

    if answer.invalid_citations:
        _report(WARN, f"cited sources that do not exist: {answer.invalid_citations}")
    if answer.covered and not answer.cited_sources:
        _report(WARN, "the answer cites no source — treat it as unverified")

    if not answer.sources:
        return

    _report(INFO, f"{len(answer.sources)} chunk(s) retrieved, {len(answer.cited_sources)} cited")
    for source in answer.sources:
        mark = "cited" if source.cited else "     "
        sys.stdout.write(f"  [{source.number}] {mark} {source.score:.3f}  {source.citation}\n")
        if show_context:
            preview = " ".join(source.text.split())[:PREVIEW_CHARS]
            sys.stdout.write(f"        {preview}...\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the policy corpus a question.")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--k", type=int, default=None, help="Chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    sys.stdout.write("Phase 5 policy RAG\n" + "-" * 70 + "\n")
    _report(INFO, f'Question: "{args.question}"')
    _report(INFO, f"model={settings.llm_model} | floor={settings.min_relevance_score} "
                  f"| ratio={settings.relative_relevance_ratio} | top_k={settings.retrieval_top_k}")
    _report(INFO, "Generating locally; the first call loads the model into RAM...")

    try:
        answer = build_policy_rag(settings).ask(args.question, k=args.k)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1

    _print_answer(answer, args.show_context)
    # A refusal is a correct outcome, not a failure: exit 0 either way, so a
    # caller cannot mistake "the corpus does not cover this" for a broken run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
