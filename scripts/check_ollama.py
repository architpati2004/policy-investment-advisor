"""Phase 2 checker: verify local inference end to end.

Confirms the Ollama server is reachable, that both configured models are pulled,
and that generation actually returns text — with timing, since first-token
latency on an M1 is the number that matters when tuning LLM_TIMEOUT_SECONDS.

Run with::

    python scripts/check_ollama.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.rag.ollama_client import check_status, generate  # noqa: E402

PASS, FAIL, INFO = "[ OK ]", "[FAIL]", "[INFO]"

SMOKE_PROMPT = "Reply with exactly one word: READY"
SMOKE_SYSTEM = "You are a terse assistant. Answer in as few words as possible."


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def main() -> int:
    settings = get_settings()
    sys.stdout.write("Phase 2 Ollama check\n" + "-" * 60 + "\n")
    _report(INFO, f"Base URL: {settings.ollama_base_url}")

    result = check_status(settings)

    if not result.reachable:
        _report(FAIL, f"Server unreachable — {result.detail}")
        _report(INFO, "Start it in another terminal: ollama serve")
        return 1
    _report(PASS, f"Server reachable ({len(result.installed_models)} models installed)")

    for name in result.installed_models:
        _report(INFO, f"  installed: {name}")

    if result.missing_models():
        for name in result.missing_models():
            _report(FAIL, f"Model not installed: {name}")
            _report(INFO, f"  Run: ollama pull {name}")
        return 1
    _report(PASS, f"Generation model present: {settings.llm_model}")
    _report(PASS, f"Embedding model present: {settings.embedding_model}")

    _report(INFO, f"Generating with {settings.llm_model} (first run loads the model into RAM)...")
    started = time.perf_counter()
    try:
        answer = generate(SMOKE_PROMPT, system=SMOKE_SYSTEM, settings=settings)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    elapsed = time.perf_counter() - started

    preview = " ".join(answer.split())[:120]
    _report(PASS, f"Generation succeeded in {elapsed:.1f}s")
    _report(INFO, f'  Model said: "{preview}"')
    if elapsed > settings.llm_timeout_seconds * 0.5:
        _report(
            INFO,
            "That is slow relative to LLM_TIMEOUT_SECONDS; consider a smaller "
            "model or a higher timeout before Phase 5.",
        )

    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write("Local inference is working. Next: Phase 3 (PDF ingestion).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
