"""Shared fixtures and test doubles.

Phases 4 through 11 each grew their own copy of the same three doubles — a
deterministic embedding model, a chat model that returns a scripted reply, and a
retriever that returns a fixed result. Four copies of ``FakeEmbeddings`` and
three of ``StubChatModel`` is three chances for them to drift out of step with
the interfaces they stand in for, and a change to either interface meant finding
every copy.

They live here now. Each is a real implementation of the interface it replaces,
not a mock: ``FakeEmbeddings`` genuinely embeds, so FAISS indexes in the tests
are real indexes, and ``StubChatModel`` streams, because that is how the chain
consumes a model since the generation deadline was added.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.ingestion.company_registry import Company, CompanyRegistry
from scripts.make_sample_pdf import write_pdf

WORD = re.compile(r"[a-z0-9]+")

#: Companies the tests hold, across four sectors so scope filtering has
#: something to discriminate between.
GODREJ = Company("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer", "gcpl"))
HDFC = Company("HDFCBANK", "HDFC Bank", "Banking")
RELIANCE = Company("RELIANCE", "Reliance Industries", "Energy")
INFY = Company("INFY", "Infosys", "IT")


@pytest.fixture
def pdf_factory(tmp_path: Path):
    """Return a callable writing a PDF into the test's temp directory.

    The writer lives in ``scripts/make_sample_pdf.py`` and is reused here, so
    tests exercise real PDFs without the project taking on a PDF-generation
    dependency.
    """

    def _make(name: str, pages: list[list[str]]) -> Path:
        return write_pdf(tmp_path / name, pages)

    return _make


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings, unit length, no network.

    Hashing each word into a fixed number of buckets means texts sharing
    vocabulary end up close together, so retrieval *ordering* can be asserted
    without a real model — while staying reproducible across runs.

    Its scores do not share embeddinggemma's scale, which is why tests about
    scoping or ordering pass an explicit floor-free ``RetrievalPolicy`` rather
    than letting the configured floors reject perfectly good matches.
    """

    dimension = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in WORD.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


class Chunk:
    """One streamed chunk, as ``ChatOllama.stream`` yields them."""

    def __init__(self, content: str) -> None:
        self.content = content


class StubChatModel:
    """Returns a scripted reply and records what it was asked.

    Streams rather than returning at once, because the chain reads generation
    chunk by chunk so a wall-clock deadline can be enforced between chunks.
    Splitting the reply in two would catch an accumulation bug.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[tuple[str, str]]] = []

    def stream(self, messages: list[tuple[str, str]]) -> Any:
        self.calls.append(messages)
        half = len(self.reply) // 2
        return iter([Chunk(self.reply[:half]), Chunk(self.reply[half:])])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temp directory.

    Database, indexes and data all move together: a test that writes to the
    project's own ``data/`` or ``backend/vectorstore/`` would corrupt the corpus
    the developer is working with.
    """
    return get_settings().model_copy(
        update={
            "database_url": f"sqlite:///{tmp_path}/test.db",
            "vectorstore_dir": str(tmp_path / "vectorstore"),
            "data_dir": str(tmp_path / "data"),
        }
    )


@pytest.fixture
def registry() -> CompanyRegistry:
    return CompanyRegistry([GODREJ, HDFC, RELIANCE, INFY])


@pytest.fixture
def db_session(settings: Settings) -> Iterator[Session]:
    """A session on a fresh database with the companies synced in."""
    from backend.db import portfolio as service
    from backend.db.session import create_db_engine, init_db

    engine = create_db_engine(settings)
    init_db(engine)
    with Session(engine, expire_on_commit=False) as session:
        service.sync_companies(session, CompanyRegistry([GODREJ, HDFC, RELIANCE, INFY]))
        yield session
    engine.dispose()


def document(text: str, **metadata: Any) -> Document:
    """A chunk-shaped document with whatever metadata a test needs."""
    return Document(page_content=text, metadata=metadata)
