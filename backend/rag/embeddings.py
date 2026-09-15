"""Embeddings from the local Ollama embedding model.

Wraps :class:`~langchain_ollama.OllamaEmbeddings` with three things the raw
client does not provide:

* **Batching** — an ingest hands over hundreds of chunks at once. Sending them
  in ``EMBEDDING_BATCH_SIZE`` groups keeps memory flat on an M1 and gives the
  ingest a progress log instead of one long silence.
* **Unit normalisation** — every vector is scaled to length 1, which turns the
  inner product used by the FAISS index into plain cosine similarity. Scores
  then live in a fixed range, so ``MIN_RELEVANCE_SCORE`` means the same thing
  for every query and every document.
* **Named errors** — the underlying client raises assorted HTTP and transport
  exceptions; callers get :class:`OllamaUnavailableError`,
  :class:`ModelNotInstalledError`, :class:`LLMTimeoutError` or
  :class:`EmbeddingError`, each carrying the command that fixes it.

The class implements LangChain's ``Embeddings`` interface, so it drops straight
into FAISS and into any retriever built in Phases 5-8.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Protocol, Sequence

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from backend.config import Settings, get_settings
from backend.exceptions import (
    EmbeddingError,
    LLMTimeoutError,
    ModelNotInstalledError,
    OllamaUnavailableError,
)
from backend.logging_config import get_logger
from backend.rag import ollama_client

logger = get_logger(__name__)

#: Text embedded when probing the model for its vector width.
_DIMENSION_PROBE = "dimension probe"


class EmbeddingClient(Protocol):
    """The slice of ``OllamaEmbeddings`` this module actually uses.

    Declared as a protocol so tests can inject a stub without a live server, and
    so swapping the backing client later touches one constructor argument.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vector: Sequence[float]) -> list[float]:
    """Scale a vector to unit length.

    A zero vector — which Ollama returns for empty or unembeddable input — is
    passed through unchanged rather than dividing by zero. It will match nothing,
    which is the right outcome for text that carried no signal.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def _translate(exc: Exception, model: str, settings: Settings) -> Exception:
    """Map a client exception onto the application's error hierarchy."""
    text = str(exc).lower()
    if "not found" in text or "try pulling" in text or "no such model" in text:
        return ModelNotInstalledError(model)
    if "timeout" in text or "timed out" in text:
        return LLMTimeoutError(model, settings.llm_timeout_seconds)
    if "connection" in text or "refused" in text or "connect" in text:
        return OllamaUnavailableError(settings.ollama_base_url, str(exc))
    return EmbeddingError(model, str(exc))


class LocalEmbeddings(Embeddings):
    """Batched, unit-normalised embeddings served by the local Ollama model.

    Construction contacts nothing; the first embedding call is what needs the
    server to be up. Call :func:`verify_ready` when you would rather find out
    before starting a long ingest.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        batch_size: int,
        timeout_seconds: int,
        settings: Settings,
        client: EmbeddingClient | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"EMBEDDING_BATCH_SIZE must be at least 1, got {batch_size}")

        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self._settings = settings
        self._dimension: int | None = None
        self._client: EmbeddingClient = client or OllamaEmbeddings(
            model=model,
            base_url=base_url,
            client_kwargs={"timeout": timeout_seconds},
        )

    # --- LangChain Embeddings interface ------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, in batches, preserving input order.

        Raises:
            OllamaUnavailableError: the server is not running.
            ModelNotInstalledError: ``EMBEDDING_MODEL`` has not been pulled.
            LLMTimeoutError: a batch exceeded the configured timeout.
            EmbeddingError: any other failure once the server was reached.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        total_batches = math.ceil(len(texts) / self.batch_size)

        for number, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            try:
                raw = self._client.embed_documents(batch)
            except Exception as exc:  # noqa: BLE001 - normalised below
                raise _translate(exc, self.model, self._settings) from exc

            if len(raw) != len(batch):
                raise EmbeddingError(
                    self.model,
                    f"expected {len(batch)} vectors for batch {number}, got {len(raw)}",
                )
            vectors.extend(_normalise(vector) for vector in raw)

            if total_batches > 1:
                logger.info(
                    "Embedded batch %d/%d (%d/%d texts)",
                    number,
                    total_batches,
                    len(vectors),
                    len(texts),
                )

        self._remember_dimension(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Kept distinct from :meth:`embed_documents` because LangChain retrievers
        call it by name, and because some embedding models prefix queries
        differently from documents.
        """
        try:
            vector = self._client.embed_query(text)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _translate(exc, self.model, self._settings) from exc

        normalised = _normalise(vector)
        self._remember_dimension(normalised)
        return normalised

    # --- Introspection ------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vector width of this model, probed once and cached.

        The index builder records it in the manifest, and the Phase 4 checker
        prints it: a model swap that changes this number invalidates every index
        on disk, so it is worth stating out loud.
        """
        if self._dimension is None:
            self._dimension = len(self.embed_query(_DIMENSION_PROBE))
        return self._dimension

    def _remember_dimension(self, vector: Sequence[float]) -> None:
        """Record the observed width, so :attr:`dimension` costs nothing later."""
        if self._dimension is None and vector:
            self._dimension = len(vector)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"LocalEmbeddings(model={self.model!r}, batch_size={self.batch_size})"


def build_embeddings(
    settings: Settings | None = None,
    *,
    client: EmbeddingClient | None = None,
    **overrides: Any,
) -> LocalEmbeddings:
    """Construct embeddings from configuration. Does not contact the server."""
    settings = settings or get_settings()
    params: dict[str, Any] = {
        "model": settings.embedding_model,
        "base_url": settings.ollama_base_url,
        "batch_size": settings.embedding_batch_size,
        "timeout_seconds": settings.llm_timeout_seconds,
        "settings": settings,
        "client": client,
    }
    params.update(overrides)
    logger.debug("Building LocalEmbeddings for model %s", params["model"])
    return LocalEmbeddings(**params)


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    """Process-wide embeddings singleton built from current settings.

    Cached because each instance holds an HTTP client and a probed dimension;
    one is enough. Use :func:`build_embeddings` when you need different
    parameters, such as a larger batch size for a bulk ingest.
    """
    return build_embeddings()


def verify_ready(settings: Settings | None = None) -> None:
    """Confirm the embedding model is pulled before a long ingest starts.

    Raises:
        OllamaUnavailableError: the server is not running.
        ModelNotInstalledError: the configured embedding model is not pulled.
    """
    settings = settings or get_settings()
    # Called through the module so tests can substitute it; ``from ... import``
    # would bind at import time.
    ollama_client.require_model(settings.embedding_model, settings)
