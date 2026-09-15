"""Typed application configuration, loaded once from ``.env``.

Every tunable value in the project — model names, directories, chunking and
retrieval parameters — is declared here and read from the environment. No other
module hard-codes a model name, URL or path; they import :func:`get_settings`.

Relative paths resolve against the project root, so the application behaves
identically no matter which directory you launch it from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

_SQLITE_PREFIX = "sqlite:///"


def _resolve(raw: str) -> Path:
    """Turn a configured path into an absolute one, anchored at the project root."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class Settings(BaseSettings):
    """Runtime settings.

    Field names map to upper-case environment variables (``llm_model`` reads
    ``LLM_MODEL``). Defaults keep a fresh clone runnable even without a ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity ----------------------------------------------------------
    app_name: str = "Policy-Based Investment Advisor"
    app_version: str = "0.1.0"

    # --- Local inference (Ollama) -----------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b"
    embedding_model: str = "embeddinggemma"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 180
    llm_num_ctx: int = 8192

    # --- Chunking and retrieval -------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    min_relevance_score: float = 0.25

    # --- Storage locations (relative paths resolve to the project root) ----
    data_dir: str = "data"
    vectorstore_dir: str = "backend/vectorstore"
    database_url: str = "sqlite:///data/app.db"

    # --- Server ------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    # --- Derived paths -----------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_path(self) -> Path:
        """Root of the raw source corpus."""
        return _resolve(self.data_dir)

    @property
    def policy_data_dir(self) -> Path:
        """RBI / SEBI / Budget / circular PDFs dropped here by the user."""
        return self.data_path / "policies"

    @property
    def company_data_dir(self) -> Path:
        """Annual reports, quarterly results, fundamentals."""
        return self.data_path / "companies"

    @property
    def news_data_dir(self) -> Path:
        """Articles fetched from RSS feeds, cached before chunking."""
        return self.data_path / "news"

    @property
    def processed_data_dir(self) -> Path:
        """Cleaned, chunk-ready intermediate artefacts."""
        return self.data_path / "processed"

    @property
    def vectorstore_path(self) -> Path:
        return _resolve(self.vectorstore_dir)

    @property
    def policy_index_dir(self) -> Path:
        """FAISS index holding regulatory and policy chunks."""
        return self.vectorstore_path / "policy_index"

    @property
    def company_index_dir(self) -> Path:
        """FAISS index holding company, fundamentals and news chunks."""
        return self.vectorstore_path / "company_index"

    @property
    def resolved_database_url(self) -> str:
        """SQLite URLs rewritten to an absolute path.

        ``sqlite:///data/app.db`` is relative to the current working directory,
        which makes the database move depending on where you start the server.
        Anchoring it to the project root avoids that whole class of bug.
        """
        if not self.database_url.startswith(_SQLITE_PREFIX):
            return self.database_url

        raw = self.database_url[len(_SQLITE_PREFIX) :]
        if raw.startswith("/"):
            return self.database_url

        absolute = _resolve(raw)
        absolute.parent.mkdir(parents=True, exist_ok=True)
        return f"{_SQLITE_PREFIX}{absolute}"

    @property
    def cors_origin_list(self) -> list[str]:
        """``CORS_ORIGINS`` is comma-separated so it stays easy to edit by hand."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # --- Helpers -----------------------------------------------------------
    def managed_directories(self) -> list[Path]:
        """Directories the application owns and may create on demand."""
        return [
            self.data_path,
            self.policy_data_dir,
            self.company_data_dir,
            self.news_data_dir,
            self.processed_data_dir,
            self.vectorstore_path,
            self.policy_index_dir,
            self.company_index_dir,
        ]

    def ensure_directories(self) -> None:
        """Create any missing data/index directories. Safe to call repeatedly."""
        for directory in self.managed_directories():
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so ``.env`` is parsed once. FastAPI routes take this via
    ``Depends(get_settings)`` instead of importing a module-level global, which
    keeps them overridable in tests.
    """
    return Settings()
