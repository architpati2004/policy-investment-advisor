"""Phase 2 tests: Ollama connection layer.

The unit tests stub the HTTP layer so they run with no Ollama installed. The
integration test at the bottom talks to a real server and skips itself when one
is not running.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.exceptions import ModelNotInstalledError, OllamaUnavailableError
from backend.main import create_app
from backend.rag import ollama_client
from backend.rag.ollama_client import (
    OllamaStatus,
    build_chat_model,
    check_status,
    list_installed_models,
    model_matches,
    require_model,
)


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def installed_models(monkeypatch: pytest.MonkeyPatch):
    """Patch the tags endpoint to report a given set of installed models."""

    def _install(names: list[str]) -> None:
        payload = {"models": [{"name": name} for name in names]}
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResponse(payload))

    return _install


# --- Model name resolution ---------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "installed", "expected"),
    [
        ("qwen3:4b", "qwen3:4b", True),
        ("qwen3", "qwen3:latest", True),
        ("qwen3", "qwen3:4b", False),
        ("qwen3:4b", "qwen3:1.7b", False),
        ("embeddinggemma", "embeddinggemma:latest", True),
        ("qwen3:4b", "llama3:8b", False),
    ],
)
def test_model_matches(configured: str, installed: str, expected: bool) -> None:
    assert model_matches(configured, installed) is expected


# --- Inspection --------------------------------------------------------------


def test_list_installed_models_returns_sorted_names(installed_models) -> None:
    installed_models(["qwen3:4b", "embeddinggemma:latest"])
    assert list_installed_models() == ["embeddinggemma:latest", "qwen3:4b"]


def test_list_installed_models_raises_when_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*args: Any, **kwargs: Any):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _refuse)
    with pytest.raises(OllamaUnavailableError) as exc:
        list_installed_models()
    assert "ollama serve" in str(exc.value)


def test_check_status_reports_ready_when_both_models_present(installed_models) -> None:
    settings = get_settings()
    installed_models([f"{settings.llm_model}", f"{settings.embedding_model}:latest"])
    result = check_status(settings)
    assert result.reachable and result.ready
    assert result.missing_models() == []


def test_check_status_lists_missing_models(installed_models) -> None:
    settings = get_settings()
    installed_models(["llama3:8b"])
    result = check_status(settings)
    assert result.reachable
    assert not result.ready
    assert set(result.missing_models()) == {settings.llm_model, settings.embedding_model}


def test_check_status_does_not_raise_when_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*args: Any, **kwargs: Any):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _refuse)
    result = check_status()
    assert result.reachable is False
    assert result.ready is False
    assert result.detail


def test_require_model_raises_with_pull_command(installed_models) -> None:
    installed_models(["llama3:8b"])
    with pytest.raises(ModelNotInstalledError) as exc:
        require_model("qwen3:4b")
    assert "ollama pull qwen3:4b" in str(exc.value)


# --- Construction ------------------------------------------------------------


def test_build_chat_model_uses_configuration() -> None:
    settings = get_settings()
    model = build_chat_model(settings)
    assert model.model == settings.llm_model
    assert model.base_url == settings.ollama_base_url
    assert model.temperature == settings.llm_temperature


def test_build_chat_model_accepts_overrides() -> None:
    assert build_chat_model(get_settings(), temperature=0.9).temperature == 0.9


# --- API ---------------------------------------------------------------------


def test_ollama_endpoint_returns_503_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(
        ollama_client,
        "check_status",
        lambda *a, **kw: OllamaStatus(
            reachable=False,
            base_url=settings.ollama_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            detail="connection refused",
        ),
    )
    with TestClient(create_app()) as client:
        response = client.get("/api/system/ollama")
    assert response.status_code == 503
    assert response.json()["reachable"] is False


def test_ollama_endpoint_returns_424_when_model_missing(installed_models) -> None:
    installed_models(["llama3:8b"])
    with TestClient(create_app()) as client:
        response = client.get("/api/system/ollama")
    assert response.status_code == 424
    assert response.json()["missing_models"]


def test_health_still_works_when_ollama_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API must stay diagnosable when local inference is unavailable."""

    def _refuse(*args: Any, **kwargs: Any):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _refuse)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


# --- Integration (skipped unless a real server is running) -------------------


def _server_is_live() -> bool:
    try:
        return check_status().reachable
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _server_is_live(), reason="Ollama is not running")
def test_live_generation_returns_text() -> None:
    from backend.rag.ollama_client import generate

    status = check_status()
    if not status.ready:
        pytest.skip(f"models not pulled: {status.missing_models()}")
    answer = generate("Reply with exactly one word: READY")
    assert isinstance(answer, str) and answer.strip()


def test_ollama_endpoint_returns_200_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the late-binding fix: patching the module must reach the route.

    Without it this test passes or fails depending on whether the developer
    happens to have Ollama running, which is exactly the kind of environment
    dependence a test suite should not have.
    """
    settings = get_settings()
    monkeypatch.setattr(
        ollama_client,
        "check_status",
        lambda *a, **kw: OllamaStatus(
            reachable=True,
            base_url=settings.ollama_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            llm_available=True,
            embedding_available=True,
            installed_models=[settings.llm_model, settings.embedding_model],
        ),
    )
    with TestClient(create_app()) as client:
        response = client.get("/api/system/ollama")
    assert response.status_code == 200
    assert response.json()["ready"] is True
