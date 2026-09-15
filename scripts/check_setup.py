"""Phase 1 setup checker.

Verifies the scaffold is sound before any feature work begins:
Python version, required directories, ``.env`` presence, importable packages,
and that configuration loads and validates.

Ollama connectivity is deliberately NOT checked here — that arrives in Phase 2.

Run with::

    python scripts/check_setup.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_DIRS: tuple[str, ...] = (
    "backend/api",
    "backend/rag",
    "backend/ingestion",
    "backend/alerts",
    "backend/db",
    "backend/vectorstore/policy_index",
    "backend/vectorstore/company_index",
    "data/policies",
    "data/companies",
    "data/news",
    "data/processed",
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/services",
    "tests",
)

CORE_PACKAGES: tuple[str, ...] = ("fastapi", "pydantic", "pydantic_settings", "uvicorn")
RAG_PACKAGES: tuple[str, ...] = (
    "langchain",
    "langchain_community",
    "langchain_ollama",
    "langchain_text_splitters",
    "faiss",
    "pypdf",
    "sqlalchemy",
    "feedparser",
    "bs4",
    "requests",
)

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    _report(PASS if ok else FAIL, f"Python {major}.{minor} (3.11+ required)")
    return ok


def check_directories() -> bool:
    missing = [d for d in REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    if missing:
        _report(FAIL, f"Missing directories: {', '.join(missing)}")
        return False
    _report(PASS, f"All {len(REQUIRED_DIRS)} project directories present")
    return True


def check_env_file() -> bool:
    if (PROJECT_ROOT / ".env").is_file():
        _report(PASS, ".env present")
        return True
    _report(FAIL, ".env missing — run: cp .env.example .env")
    return False


def check_packages(packages: tuple[str, ...], label: str, required: bool) -> bool:
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if not missing:
        _report(PASS, f"{label} packages importable")
        return True
    status = FAIL if required else WARN
    _report(status, f"{label} packages missing: {', '.join(missing)}")
    return not required


def check_settings() -> bool:
    try:
        from backend.config import get_settings

        settings = get_settings()
        settings.ensure_directories()
    except Exception as exc:  # noqa: BLE001 - surface any config error to the user
        _report(FAIL, f"Configuration failed to load: {exc}")
        return False

    _report(PASS, f"Configuration loaded (llm={settings.llm_model}, embeddings={settings.embedding_model})")
    _report(PASS, f"Database URL: {settings.resolved_database_url}")
    _report(PASS, f"Policy index dir: {settings.policy_index_dir}")
    _report(PASS, f"Company index dir: {settings.company_index_dir}")
    return True


def main() -> int:
    sys.stdout.write("Phase 1 setup check\n" + "-" * 60 + "\n")
    results = [
        check_python(),
        check_directories(),
        check_env_file(),
        check_packages(CORE_PACKAGES, "Core", required=True),
        check_packages(RAG_PACKAGES, "RAG/ingestion", required=False),
        check_settings(),
    ]
    sys.stdout.write("-" * 60 + "\n")
    if all(results):
        sys.stdout.write("Phase 1 scaffold is ready. Next: Phase 2 (Ollama connection).\n")
        return 0
    sys.stdout.write("Some checks failed — see above.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
