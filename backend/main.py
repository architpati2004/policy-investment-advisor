"""FastAPI application entrypoint.

Mounts the system router from Phase 2 and the feature routers from Phase 11:
portfolio, chat, documents and alerts.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import alerts, chat, documents, portfolio, system
from backend.config import get_settings
from backend.exceptions import AdvisorError
from backend.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create local directories on startup and log the active configuration.

    Ollama is reported on but never required here: the API must start even when
    the model server is down, so ``/api/system/ollama`` can explain why.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_directories()

    from backend.db.session import init_db

    # Idempotent, and cheap: the API should serve a portfolio request on a
    # fresh checkout without a separate setup step.
    init_db(settings=settings)
    logger.info(
        "%s v%s ready | llm=%s | embeddings=%s | ollama=%s",
        settings.app_name,
        settings.app_version,
        settings.llm_model,
        settings.embedding_model,
        settings.ollama_base_url,
    )

    from backend.rag.ollama_client import check_status

    ollama = check_status(settings)
    if ollama.ready:
        logger.info("Local inference ready (%d models installed)", len(ollama.installed_models))
    elif not ollama.reachable:
        logger.warning("Ollama unreachable at %s — start it with 'ollama serve'", ollama.base_url)
    else:
        logger.warning("Ollama up but models missing: %s", ", ".join(ollama.missing_models()))

    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build the application. A factory, so tests can spin up isolated instances."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Retrieval-augmented research assistant that reads regulatory policy "
            "and company news against a user's portfolio. Runs entirely locally."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AdvisorError)
    async def advisor_error_handler(request: Request, exc: AdvisorError) -> JSONResponse:
        """Turn any application error into a response carrying its own fix hint."""
        logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    app.include_router(system.router)
    app.include_router(portfolio.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(alerts.router)
    return app


app = create_app()
