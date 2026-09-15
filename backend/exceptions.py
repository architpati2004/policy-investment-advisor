"""Application exception hierarchy.

Every failure mode listed in the project brief gets a named exception carrying a
message that tells the user how to fix it. API routes translate these into HTTP
responses in Phase 11; until then they surface in the CLI checkers.
"""

from __future__ import annotations


class AdvisorError(Exception):
    """Base class for all application errors.

    ``status_code`` is the HTTP status a route should return for this failure.
    Keeping it on the exception means route handlers do not need a lookup table.
    """

    status_code: int = 500

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation

    def __str__(self) -> str:
        if self.remediation:
            return f"{self.message}\n  Fix: {self.remediation}"
        return self.message

    def to_dict(self) -> dict[str, str | None]:
        """Serialisable form for API error responses."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "remediation": self.remediation,
        }


# --- Local inference ---------------------------------------------------------


class OllamaError(AdvisorError):
    """Base for anything wrong with the local Ollama server."""

    status_code = 503


class OllamaUnavailableError(OllamaError):
    """The Ollama server is not reachable at the configured base URL."""

    def __init__(self, base_url: str, detail: str | None = None) -> None:
        message = f"Cannot reach Ollama at {base_url}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(
            message,
            remediation="Start the server with 'ollama serve', then retry.",
        )
        self.base_url = base_url


class ModelNotInstalledError(OllamaError):
    """A configured model has not been pulled onto this machine."""

    status_code = 424

    def __init__(self, model: str, installed: list[str] | None = None) -> None:
        available = ", ".join(installed) if installed else "none"
        super().__init__(
            f"Model '{model}' is not installed (installed: {available})",
            remediation=f"Run: ollama pull {model}",
        )
        self.model = model
        self.installed = installed or []


class LLMTimeoutError(OllamaError):
    """Generation exceeded ``LLM_TIMEOUT_SECONDS``."""

    status_code = 504

    def __init__(self, model: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Model '{model}' did not respond within {timeout_seconds:g}s",
            remediation=(
                "Local inference is slow on first load. Raise LLM_TIMEOUT_SECONDS "
                "in .env, or switch LLM_MODEL to a smaller model such as qwen3:1.7b."
            ),
        )
        self.model = model
        self.timeout_seconds = timeout_seconds


class LLMGenerationError(OllamaError):
    """The model was reached but generation failed."""

    def __init__(self, model: str, detail: str) -> None:
        super().__init__(
            f"Generation failed for model '{model}': {detail}",
            remediation="Check the 'ollama serve' terminal for the underlying error.",
        )
        self.model = model


# --- Document ingestion ------------------------------------------------------


class DocumentError(AdvisorError):
    """Base for anything wrong with a source document."""

    status_code = 422


class InvalidDocumentError(DocumentError):
    """The file is missing, the wrong type, or corrupt."""

    def __init__(self, path: str, detail: str) -> None:
        super().__init__(
            f"Cannot read document '{path}': {detail}",
            remediation="Check the file opens in a PDF viewer and re-download it if not.",
        )
        self.path = path


class EncryptedDocumentError(DocumentError):
    """The PDF is password-protected and cannot be read."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Document '{path}' is password-protected",
            remediation="Open it in Preview and re-save without a password, then re-ingest.",
        )
        self.path = path


class EmptyDocumentError(DocumentError):
    """The document parsed but yielded no usable text."""

    def __init__(self, path: str, detail: str) -> None:
        super().__init__(
            f"No text extracted from '{path}': {detail}",
            remediation=(
                "Scanned PDFs hold images, not text. Run OCR (for example "
                "'ocrmypdf in.pdf out.pdf') and ingest the output."
            ),
        )
        self.path = path


# --- Embeddings --------------------------------------------------------------


class EmbeddingError(OllamaError):
    """The embedding model was reached but embedding failed."""

    def __init__(self, model: str, detail: str) -> None:
        super().__init__(
            f"Embedding failed for model '{model}': {detail}",
            remediation="Check the 'ollama serve' terminal for the underlying error.",
        )
        self.model = model


# --- Vector store ------------------------------------------------------------


class VectorStoreError(AdvisorError):
    """Base for anything wrong with a FAISS index on disk."""

    status_code = 500


class IndexNotFoundError(VectorStoreError):
    """A query arrived before the index it needs was built."""

    status_code = 404

    def __init__(self, name: str, directory: str) -> None:
        super().__init__(
            f"The {name} index has not been built yet (looked in {directory})",
            remediation="Run: python scripts/build_policy_index.py build",
        )
        self.name = name
        self.directory = directory


class IndexModelMismatchError(VectorStoreError):
    """The index on disk was written by a different embedding model.

    Vectors from two different models share no geometry, so searching an index
    with the wrong model returns confident nonsense rather than an obvious
    failure. Refusing to load is the only safe response.
    """

    status_code = 409

    def __init__(self, name: str, indexed_model: str, configured_model: str) -> None:
        super().__init__(
            f"The {name} index was built with embedding model '{indexed_model}', "
            f"but EMBEDDING_MODEL is now '{configured_model}'",
            remediation=(
                "Rebuild the index with the current model: "
                "python scripts/build_policy_index.py build --rebuild"
            ),
        )
        self.name = name
        self.indexed_model = indexed_model
        self.configured_model = configured_model


class IndexCorruptError(VectorStoreError):
    """The index files exist but cannot be read back."""

    status_code = 500

    def __init__(self, name: str, directory: str, detail: str) -> None:
        super().__init__(
            f"The {name} index at {directory} could not be loaded: {detail}",
            remediation=(
                "Delete the directory's index files and rebuild: "
                "python scripts/build_policy_index.py build --rebuild"
            ),
        )
        self.name = name
        self.directory = directory


# --- Company data ------------------------------------------------------------


class InvalidRegistryError(DocumentError):
    """The company registry is malformed."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"Company registry is invalid: {detail}",
            remediation=(
                "Fix data/companies/registry.json — a JSON list of objects with "
                "'ticker', 'name', 'sector' and optional 'aliases'."
            ),
        )


class UnknownCompanyError(DocumentError):
    """A company document cannot be attributed to a company.

    Indexing it anyway would put a chunk in the company index that no holding
    can ever match, and that nothing can honestly cite.
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Cannot tell which company '{path}' belongs to",
            remediation=(
                "Add it to data/companies/registry.json (ticker, name, sector, "
                "aliases), or pass --company and --sector on the command line."
            ),
        )
        self.path = path


# --- Portfolio ---------------------------------------------------------------


class PortfolioError(AdvisorError):
    """Something is wrong with a portfolio or an operation on it."""

    status_code = 400


class UnknownTickerError(PortfolioError):
    """A holding names a company the system has never heard of.

    Refused rather than stored: a holding whose ticker matches no company has
    no sector, so it can never be matched to a policy change, and it would sit
    in the portfolio looking as though it were being watched.
    """

    status_code = 404

    def __init__(self, ticker: str, known: list[str] | None = None) -> None:
        available = ", ".join(known) if known else "none"
        super().__init__(
            f"'{ticker}' is not a known company (known: {available})",
            remediation=(
                "Declare it in data/companies/registry.json, then run: "
                "python scripts/portfolio.py sync"
            ),
        )
        self.ticker = ticker
        self.known = known or []
