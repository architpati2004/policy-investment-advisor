"""Shared FastAPI dependencies.

Every chain and engine is built through a dependency rather than imported as a
module global, for the reason CLAUDE.md section 10 gives about monkeypatching:
a route that constructs its own chain cannot be tested without a live model, and
a route that reads a global cannot be tested twice with different settings.

Construction is cheap — chains build their chat model lazily and indexes load on
first use — so a per-request dependency costs nothing and keeps the wiring
visible in the signature.
"""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.alerts.engine import AlertEngine, build_engine
from backend.config import Settings, get_settings
from backend.db.session import get_session_factory
from backend.rag.policy_rag import PolicyRAG, build_policy_rag
from backend.rag.portfolio_rag import PortfolioRAG, build_portfolio_rag
from backend.rag.vector_store import VectorIndex, company_index, news_index, policy_index


def get_db(settings: Settings = Depends(get_settings)) -> Iterator[Session]:
    """A database session per request, closed afterwards.

    Reads the engine from the process-wide factory built in Phase 7, so the
    connection and its foreign-key pragma are configured in exactly one place.
    """
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_policy_rag(settings: Settings = Depends(get_settings)) -> PolicyRAG:
    return build_policy_rag(settings)


def get_portfolio_rag(settings: Settings = Depends(get_settings)) -> PortfolioRAG:
    return build_portfolio_rag(settings)


def get_alert_engine(settings: Settings = Depends(get_settings)) -> AlertEngine:
    return build_engine(settings)


#: The three corpora, by the name a client addresses them with.
INDEX_FACTORIES = {"policy": policy_index, "company": company_index, "news": news_index}


def get_indexes(settings: Settings = Depends(get_settings)) -> dict[str, VectorIndex]:
    """Every index, addressable by name.

    A dependency rather than a module lookup so a test can hand the routes an
    index built with deterministic embeddings. Without that, a search test has
    to embed its query through Ollama to match whatever built the index, which
    makes retrieval untestable without a running model — and the first version
    of this route had exactly that problem.

    Construction is free: a VectorIndex touches neither disk nor network until
    it is searched.
    """
    return {name: factory(settings=settings) for name, factory in INDEX_FACTORIES.items()}
