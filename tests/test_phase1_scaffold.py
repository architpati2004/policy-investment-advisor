"""Phase 1 smoke tests: configuration loads, directories exist, app boots."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT, get_settings
from backend.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_settings_load_with_expected_defaults() -> None:
    settings = get_settings()
    assert settings.llm_model
    assert settings.embedding_model
    assert settings.ollama_base_url.startswith("http")
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.retrieval_top_k > 0


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()


def test_index_directories_are_distinct_and_absolute() -> None:
    settings = get_settings()
    assert settings.policy_index_dir != settings.company_index_dir
    assert settings.policy_index_dir.is_absolute()
    assert settings.company_index_dir.is_absolute()


def test_database_url_is_resolved_to_absolute_path() -> None:
    settings = get_settings()
    assert settings.resolved_database_url.startswith("sqlite:////") or not settings.resolved_database_url.startswith(
        "sqlite:///data"
    )


def test_ensure_directories_creates_expected_tree() -> None:
    settings = get_settings()
    settings.ensure_directories()
    for directory in (
        settings.policy_data_dir,
        settings.company_data_dir,
        settings.news_data_dir,
        settings.processed_data_dir,
        settings.policy_index_dir,
        settings.company_index_dir,
    ):
        assert directory.is_dir(), f"{directory} was not created"


@pytest.mark.parametrize(
    "package_path",
    ["backend/api", "backend/rag", "backend/ingestion", "backend/alerts", "backend/db"],
)
def test_backend_packages_are_importable_packages(package_path: str) -> None:
    assert (PROJECT_ROOT / Path(package_path) / "__init__.py").is_file()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_exposes_no_secrets(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_model"] == get_settings().llm_model
    assert not any("key" in field.lower() or "secret" in field.lower() for field in payload)
