"""Connection layer for the local Ollama server.

Two responsibilities, kept apart on purpose:

* **Inspection** — plain HTTP against Ollama's ``/api/tags`` to answer "is the
  server up?" and "which models are pulled?". No LangChain involved, so the
  health check keeps working even if a LangChain import breaks.
* **Construction** — build a configured :class:`~langchain_ollama.ChatOllama`
  for the rest of the ``rag`` package to use.

Everything reads from :class:`~backend.config.Settings`; no model name or URL is
written into this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import requests
from langchain_ollama import ChatOllama

from backend.config import Settings, get_settings
from backend.exceptions import (
    AdvisorError,
    LLMGenerationError,
    LLMTimeoutError,
    ModelNotInstalledError,
    OllamaUnavailableError,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)

#: Short timeout for inspection calls. Generation uses ``LLM_TIMEOUT_SECONDS``.
PROBE_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class OllamaStatus:
    """Snapshot of the local inference environment."""

    reachable: bool
    base_url: str
    llm_model: str
    embedding_model: str
    llm_available: bool = False
    embedding_available: bool = False
    installed_models: list[str] = field(default_factory=list)
    detail: str | None = None

    @property
    def ready(self) -> bool:
        """True when the server is up and both configured models are pulled."""
        return self.reachable and self.llm_available and self.embedding_available

    def missing_models(self) -> list[str]:
        """Models named in configuration that still need pulling."""
        missing = []
        if self.reachable and not self.llm_available:
            missing.append(self.llm_model)
        if self.reachable and not self.embedding_available:
            missing.append(self.embedding_model)
        return missing

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the ``/api/system/ollama`` endpoint."""
        return {
            "reachable": self.reachable,
            "ready": self.ready,
            "base_url": self.base_url,
            "llm_model": self.llm_model,
            "llm_available": self.llm_available,
            "embedding_model": self.embedding_model,
            "embedding_available": self.embedding_available,
            "installed_models": self.installed_models,
            "missing_models": self.missing_models(),
            "detail": self.detail,
        }


def model_matches(configured: str, installed: str) -> bool:
    """Compare a configured model name against an installed one.

    Ollama stores models as ``name:tag``. A configured name without a tag should
    match its ``:latest`` build, so ``qwen3`` matches ``qwen3:latest`` but not
    ``qwen3:4b`` — mirroring how ``ollama run`` resolves names.
    """
    configured = configured.strip()
    if configured == installed:
        return True
    if ":" not in configured:
        return installed == f"{configured}:latest"
    return False


def list_installed_models(settings: Settings | None = None) -> list[str]:
    """Return the model names pulled onto this machine.

    Raises:
        OllamaUnavailableError: if the server cannot be reached or replies badly.
    """
    settings = settings or get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    try:
        response = requests.get(url, timeout=PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(settings.ollama_base_url, "connection refused") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(
            settings.ollama_base_url, f"no response within {PROBE_TIMEOUT_SECONDS}s"
        ) from exc
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise OllamaUnavailableError(settings.ollama_base_url, str(exc)) from exc

    models = payload.get("models") or []
    return sorted(entry["name"] for entry in models if isinstance(entry, dict) and entry.get("name"))


def check_status(settings: Settings | None = None) -> OllamaStatus:
    """Inspect the local environment without raising.

    Returns a status object describing what is and is not ready, so callers can
    report every problem at once rather than failing on the first one.
    """
    settings = settings or get_settings()

    try:
        installed = list_installed_models(settings)
    except OllamaUnavailableError as exc:
        logger.warning("Ollama unreachable: %s", exc.message)
        return OllamaStatus(
            reachable=False,
            base_url=settings.ollama_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            detail=exc.message,
        )

    return OllamaStatus(
        reachable=True,
        base_url=settings.ollama_base_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        llm_available=any(model_matches(settings.llm_model, name) for name in installed),
        embedding_available=any(model_matches(settings.embedding_model, name) for name in installed),
        installed_models=installed,
    )


def require_model(model: str, settings: Settings | None = None) -> None:
    """Assert a model is pulled and ready to use.

    Raises:
        OllamaUnavailableError: server not running.
        ModelNotInstalledError: server running but the model is not pulled.
    """
    settings = settings or get_settings()
    installed = list_installed_models(settings)
    if not any(model_matches(model, name) for name in installed):
        raise ModelNotInstalledError(model, installed)


def build_chat_model(settings: Settings | None = None, **overrides: Any) -> ChatOllama:
    """Construct a ChatOllama bound to the configured local model.

    Does not contact the server; call :func:`require_model` first if you need the
    model's presence confirmed up front. ``overrides`` (e.g. ``temperature=0``)
    are passed through for callers with different needs, such as the alert engine.
    """
    settings = settings or get_settings()
    params: dict[str, Any] = {
        "model": settings.llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.llm_temperature,
        "num_ctx": settings.llm_num_ctx,
        "client_kwargs": {"timeout": settings.llm_timeout_seconds},
    }
    # Left at the model's own default unless configured. Measured on qwen3:4b:
    # disabling it does not stop the model reasoning, it stops Ollama separating
    # the reasoning out, so the chain of thought lands in the answer instead.
    if settings.llm_reasoning is not None:
        params["reasoning"] = settings.llm_reasoning
    params.update(overrides)
    logger.debug("Building ChatOllama with %s", {k: v for k, v in params.items() if k != "client_kwargs"})
    return ChatOllama(**params)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """Process-wide ChatOllama singleton built from current settings.

    Cached because each instance holds an HTTP client; one is enough. Use
    :func:`build_chat_model` directly when you need non-default parameters.
    """
    return build_chat_model()


def invoke_messages(
    model: ChatOllama,
    messages: list[tuple[str, str]],
    settings: Settings | None = None,
    *,
    timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Invoke a chat model and return its text, under a real time limit.

    Shared by :func:`generate` and the RAG chains so that "Ollama died halfway
    through" reads the same wherever it happens.

    **Why this consumes a stream rather than calling ``invoke``.**
    ``ChatOllama.invoke`` streams internally and aggregates, and the timeout in
    ``client_kwargs`` is an httpx timeout — which bounds a single read, meaning
    the gap between two tokens. Tokens arrive every few tens of milliseconds, so
    that timeout never fires however long generation runs: a 227s generation
    sailed past a 180s setting, because no individual gap came close. Consuming
    the stream here puts a wall clock on the whole call, and abandoning the
    stream disconnects the client, which is what tells Ollama to stop
    generating. The httpx timeout still does its own job underneath: it catches
    a server that goes quiet and never sends a first token at all.

    Args:
        model: Chat model to run.
        messages: ``(role, content)`` pairs.
        settings: Source of the model name and default timeout.
        timeout_seconds: Overrides ``LLM_TIMEOUT_SECONDS`` for this call.
        clock: Monotonic time source; injectable so tests need not really wait.

    Raises:
        LLMTimeoutError: generation exceeded the deadline.
        OllamaUnavailableError: the server went away mid-request.
        LLMGenerationError: any other failure once the server was reached.
    """
    settings = settings or get_settings()
    deadline = settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
    started = clock()
    parts: list[str] = []
    stream = None

    try:
        stream = model.stream(messages)
        for chunk in stream:
            content = getattr(chunk, "content", "")
            parts.append(content if isinstance(content, str) else str(content))
            elapsed = clock() - started
            if elapsed > deadline:
                logger.warning(
                    "Abandoning generation after %.1fs (limit %gs); %d chunks received",
                    elapsed,
                    deadline,
                    len(parts),
                )
                raise LLMTimeoutError(settings.llm_model, deadline)
    except AdvisorError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise varied client errors
        text = str(exc).lower()
        if "timeout" in text or "timed out" in text:
            raise LLMTimeoutError(settings.llm_model, deadline) from exc
        if "connection" in text or "refused" in text:
            raise OllamaUnavailableError(settings.ollama_base_url, str(exc)) from exc
        raise LLMGenerationError(settings.llm_model, str(exc)) from exc
    finally:
        # Closing the generator drops the HTTP connection, so an abandoned
        # generation stops costing CPU instead of running on unattended.
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    return "".join(parts)


def generate(prompt: str, system: str | None = None, settings: Settings | None = None) -> str:
    """Run a single prompt through the local model and return the text.

    A thin convenience wrapper used by the Phase 2 checker and smoke tests. The
    real RAG chains in Phases 5-8 use LangChain composition instead.

    Raises:
        LLMTimeoutError: generation exceeded the configured timeout.
        LLMGenerationError: any other failure once the server was reached.
    """
    settings = settings or get_settings()
    messages: list[tuple[str, str]] = []
    if system:
        messages.append(("system", system))
    messages.append(("human", prompt))

    return invoke_messages(build_chat_model(settings), messages, settings)
