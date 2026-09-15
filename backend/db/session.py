"""Database engine and session handling.

Kept apart from the models so that tests, the CLI and the Phase 11 API can each
get a session their own way without importing each other's assumptions.

Two SQLite details are handled here rather than being left to surprise someone:
foreign keys are enforced (SQLite ignores them unless asked), and the path comes
from ``resolved_database_url``, which anchors a relative ``sqlite:///data/app.db``
to the project root so the database does not move with the working directory.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.config import Settings, get_settings
from backend.db.models import Base
from backend.logging_config import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    """Turn on foreign key enforcement for a SQLite connection.

    SQLite accepts ``REFERENCES`` clauses and then ignores them unless this
    pragma is set per connection, which would let a holding reference a company
    that does not exist — exactly the orphan the schema is written to prevent.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_engine(settings: Settings | None = None, **kwargs: Any) -> Engine:
    """Build an engine from configuration. Creates no tables."""
    settings = settings or get_settings()
    url = settings.resolved_database_url
    engine = create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_foreign_keys)
    logger.debug("Database engine for %s", url)
    return engine


def get_engine(settings: Settings | None = None) -> Engine:
    """Process-wide engine, built once."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_db_engine(settings)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Session factory bound to the process-wide engine."""
    get_engine(settings)
    assert _session_factory is not None
    return _session_factory


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> Engine:
    """Create any missing tables. Safe to run repeatedly.

    No migration framework: the schema is small and the database is local and
    rebuildable. If the schema ever has to change under real data, that is the
    point to add Alembic rather than to improvise.
    """
    engine = engine or get_engine(settings)
    Base.metadata.create_all(engine)
    logger.info("Database ready at %s", engine.url)
    return engine


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """A transactional session: commits on success, rolls back on failure."""
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session per request (used from Phase 11)."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine, so a test can point at a different database."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
