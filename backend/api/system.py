"""System and diagnostic endpoints.

Kept separate from the feature routers added in Phase 11 so that "is the stack
alive?" never depends on the RAG, database or ingestion layers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from backend.config import Settings, get_settings
from backend.rag import ollama_client

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. No dependencies, so it answers even when Ollama is down."""
    return {"status": "ok"}


@router.get("/api/config")
def config_summary(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Non-secret configuration, for debugging a local setup from the browser."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "ollama_base_url": settings.ollama_base_url,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_top_k": settings.retrieval_top_k,
        "policy_index_built": (settings.policy_index_dir / "index.faiss").is_file(),
        "company_index_built": (settings.company_index_dir / "index.faiss").is_file(),
    }


@router.get("/api/system/ollama")
def ollama_status(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Report whether local inference is ready.

    Returns 200 when the server is up and both configured models are pulled,
    503 when the server is unreachable, and 424 when it is up but a model is
    missing — the caller can tell "start Ollama" apart from "pull a model"
    without parsing the message.
    """
    # Called through the module, not a direct import, so the symbol resolves at
    # call time. A `from ... import check_status` binds at import time and cannot
    # be substituted in tests.
    result = ollama_client.check_status(settings)
    if not result.reachable:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not result.ready:
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
    return result.to_dict()
