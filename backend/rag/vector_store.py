"""FAISS index persistence, deduplicating ingest and scored retrieval.

One class, :class:`VectorIndex`, serves both indexes the architecture calls for:
the policy index built here in Phase 4 and the company index built in Phase 6.
They differ only in directory and name, so they share an implementation rather
than a copy of one.

Three decisions worth stating:

* **Cosine similarity, not raw L2.** Vectors arrive unit-normalised from
  :mod:`backend.rag.embeddings`, and the index is an inner-product one, so the
  score FAISS returns *is* the cosine similarity between query and chunk: 1.0 is
  identical, 0.0 unrelated. That makes ``MIN_RELEVANCE_SCORE`` a meaningful
  constant instead of a number tuned per corpus, and it makes scores printable
  in citations. The distance strategy is not stored in the FAISS files, so it is
  re-applied on every load — an index reloaded without it would silently switch
  to Euclidean and reorder results.
* **``chunk_id`` is the docstore key.** Re-ingesting an unchanged document
  produces identical ids (see :func:`backend.ingestion.chunker.chunk_id`), so
  ingest can skip what is already indexed instead of growing the index on every
  run.
* **A manifest beside the index.** Vectors from two embedding models share no
  geometry; searching an index with the wrong model returns confident nonsense
  rather than an error. The manifest records which model wrote the index, and
  loading refuses when it no longer matches configuration.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import Settings, get_settings
from backend.exceptions import (
    IndexCorruptError,
    IndexModelMismatchError,
    IndexNotFoundError,
)
from backend.ingestion.chunker import chunk_id as compute_chunk_id
from backend.logging_config import get_logger
from backend.rag.embeddings import build_embeddings

logger = get_logger(__name__)

#: Base name FAISS uses for its two files: ``index.faiss`` and ``index.pkl``.
INDEX_NAME = "index"
MANIFEST_FILENAME = "manifest.json"
#: Unit-length vectors plus an inner-product index gives cosine similarity.
DISTANCE_STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT
#: Over-fetch when a metadata filter is in play, so filtering cannot starve top-k.
FILTER_FETCH_MULTIPLIER = 4


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and how well it matched.

    ``score`` is cosine similarity in ``[-1.0, 1.0]``; in practice embedding
    models return values between roughly 0.1 and 0.9.
    """

    document: Document
    score: float

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    def citation(self) -> str:
        """Short human-readable provenance, e.g. ``RBI — Master Circular (p. 4)``."""
        meta = self.metadata
        parts = [str(meta.get("source", "unknown"))]
        title = meta.get("title")
        if title:
            parts.append(str(title))
        reference = " — ".join(parts)
        page = meta.get("page")
        if page is not None:
            reference = f"{reference} (p. {page})"
        date = meta.get("date")
        return f"{reference}, {date}" if date else reference

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for API responses and the CLI."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "citation": self.citation(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AddReport:
    """Outcome of adding chunks to an index."""

    added: int
    skipped: int
    elapsed_seconds: float

    @property
    def total(self) -> int:
        return self.added + self.skipped


class VectorIndex:
    """A FAISS index on disk, with deduplicating ingest and scored search.

    The store is loaded lazily: constructing this object touches neither disk
    nor network, so a FastAPI dependency can hold one whether or not the index
    has been built yet.
    """

    def __init__(
        self,
        directory: str | Path,
        embeddings: Embeddings,
        *,
        name: str,
        settings: Settings | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.embeddings = embeddings
        self.name = name
        self._settings = settings or get_settings()
        self._store: FAISS | None = None

    # --- Location on disk ---------------------------------------------------

    @property
    def index_file(self) -> Path:
        return self.directory / f"{INDEX_NAME}.faiss"

    @property
    def manifest_file(self) -> Path:
        return self.directory / MANIFEST_FILENAME

    @property
    def exists(self) -> bool:
        """True when a built index is present on disk."""
        return self.index_file.is_file() and (self.directory / f"{INDEX_NAME}.pkl").is_file()

    @property
    def loaded(self) -> bool:
        """True when the index is in memory, built or loaded."""
        return self._store is not None

    # --- Loading and saving -------------------------------------------------

    def load(self) -> FAISS:
        """Read the index from disk into memory.

        Raises:
            IndexNotFoundError: nothing has been built in this directory.
            IndexModelMismatchError: built by a different embedding model.
            IndexCorruptError: the files are present but unreadable.
        """
        if self._store is not None:
            return self._store
        if not self.exists:
            raise IndexNotFoundError(self.name, str(self.directory))

        self._check_manifest()

        try:
            store = FAISS.load_local(
                str(self.directory),
                self.embeddings,
                INDEX_NAME,
                # The files are written by this application into a local
                # directory it owns; nothing here came off the network.
                allow_dangerous_deserialization=True,
                distance_strategy=DISTANCE_STRATEGY,
            )
        except Exception as exc:  # noqa: BLE001 - pickle/faiss raise assorted types
            raise IndexCorruptError(self.name, str(self.directory), str(exc)) from exc

        self._store = store
        logger.info("Loaded %s index: %d vectors from %s", self.name, self.count(), self.directory)
        return store

    def _check_manifest(self) -> None:
        """Refuse an index written by a different embedding model."""
        manifest = self.manifest()
        if not manifest:
            logger.warning(
                "No manifest beside the %s index; cannot confirm it was built with '%s'",
                self.name,
                self._settings.embedding_model,
            )
            return

        indexed_model = manifest.get("embedding_model")
        if indexed_model and indexed_model != self._settings.embedding_model:
            raise IndexModelMismatchError(self.name, indexed_model, self._settings.embedding_model)

    def manifest(self) -> dict[str, Any]:
        """Read the sidecar manifest, or ``{}`` when absent or unreadable."""
        if not self.manifest_file.is_file():
            return {}
        try:
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable manifest at %s: %s", self.manifest_file, exc)
            return {}

    def save(self) -> None:
        """Write the index and its manifest to disk.

        Saving an index that holds nothing is a no-op: an empty directory is a
        clearer "not built yet" signal than a zero-vector index that loads fine
        and answers nothing.
        """
        if self._store is None:
            logger.debug("Nothing to save for the %s index", self.name)
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.directory), INDEX_NAME)
        self._write_manifest()
        logger.info("Saved %s index: %d vectors to %s", self.name, self.count(), self.directory)

    def _write_manifest(self) -> None:
        manifest = {
            "name": self.name,
            "embedding_model": self._settings.embedding_model,
            "dimension": self.dimension(),
            "vector_count": self.count(),
            "distance": "cosine",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def clear(self) -> None:
        """Delete the index from memory and disk. Used by ``--rebuild``."""
        self._store = None
        for path in (
            self.index_file,
            self.directory / f"{INDEX_NAME}.pkl",
            self.manifest_file,
        ):
            path.unlink(missing_ok=True)
        logger.info("Cleared the %s index at %s", self.name, self.directory)

    # --- Ingest -------------------------------------------------------------

    def chunk_ids(self) -> set[str]:
        """Ids already in the index. Empty when nothing has been built."""
        if self._store is None:
            if not self.exists:
                return set()
            self.load()
        assert self._store is not None
        return set(self._store.index_to_docstore_id.values())

    def add_documents(self, chunks: list[Document], *, save: bool = True) -> AddReport:
        """Add chunks, skipping any whose ``chunk_id`` is already indexed.

        Args:
            chunks: Chunked documents from :mod:`backend.ingestion.chunker`.
            save: Persist to disk afterwards. ``False`` lets a caller add in
                several passes and save once.

        Returns:
            A report of how many chunks were added and how many were already
            present — the number an ingest run should print.
        """
        started = time.perf_counter()
        if not chunks:
            return AddReport(added=0, skipped=0, elapsed_seconds=0.0)

        known = self.chunk_ids()
        fresh: list[Document] = []
        ids: list[str] = []

        for chunk in chunks:
            identifier = chunk.metadata.get("chunk_id") or compute_chunk_id(
                chunk.page_content, chunk.metadata
            )
            # Skip ids already on disk *and* repeats inside this call: FAISS
            # rejects a batch containing the same id twice.
            if identifier in known:
                continue
            known.add(identifier)
            fresh.append(chunk)
            ids.append(identifier)

        skipped = len(chunks) - len(fresh)
        if not fresh:
            logger.info("%s index unchanged: all %d chunks already indexed", self.name, skipped)
            return AddReport(added=0, skipped=skipped, elapsed_seconds=time.perf_counter() - started)

        logger.info("Embedding %d new chunks for the %s index", len(fresh), self.name)
        texts = [chunk.page_content for chunk in fresh]
        metadatas = [dict(chunk.metadata) for chunk in fresh]

        if self._store is None:
            self._store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas,
                ids=ids,
                distance_strategy=DISTANCE_STRATEGY,
            )
        else:
            self._store.add_texts(texts, metadatas=metadatas, ids=ids)

        if save:
            self.save()

        elapsed = time.perf_counter() - started
        logger.info(
            "%s index: added %d chunks, skipped %d duplicates in %.1fs",
            self.name,
            len(fresh),
            skipped,
            elapsed,
        )
        return AddReport(added=len(fresh), skipped=skipped, elapsed_seconds=elapsed)

    # --- Retrieval ----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        min_score: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the closest chunks to ``query``, best match first.

        Args:
            query: Natural-language question.
            k: How many results to return; defaults to ``RETRIEVAL_TOP_K``.
            min_score: Drop results below this cosine similarity; defaults to
                ``MIN_RELEVANCE_SCORE``. Pass ``0.0`` to see everything, which
                is what the CLI does when tuning the threshold.
            filter: Metadata equality filter, e.g. ``{"source": "RBI"}``.
                Phase 8 uses it to restrict retrieval to a holding's sector.

        Raises:
            IndexNotFoundError: the index has not been built yet.
        """
        store = self.load()
        k = self._settings.retrieval_top_k if k is None else k
        threshold = self._settings.min_relevance_score if min_score is None else min_score
        if k < 1:
            return []

        matches = store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            fetch_k=max(k * FILTER_FETCH_MULTIPLIER, k),
        )

        results = [
            SearchResult(document=document, score=float(score))
            for document, score in matches
            if float(score) >= threshold
        ]
        logger.debug(
            "%s index: query returned %d/%d results above %.2f",
            self.name,
            len(results),
            len(matches),
            threshold,
        )
        return results

    def as_retriever(self, **kwargs: Any):
        """LangChain retriever over this index, for the chains built in Phase 5."""
        store = self.load()
        search_kwargs = {"k": self._settings.retrieval_top_k, **kwargs.pop("search_kwargs", {})}
        return store.as_retriever(search_kwargs=search_kwargs, **kwargs)

    # --- Introspection ------------------------------------------------------

    def count(self) -> int:
        """Number of vectors held in memory. Zero when nothing is loaded."""
        return 0 if self._store is None else int(self._store.index.ntotal)

    def dimension(self) -> int | None:
        """Vector width of the loaded index, or ``None`` when nothing is loaded."""
        return None if self._store is None else int(self._store.index.d)

    def documents(self) -> Iterator[Document]:
        """Iterate every indexed chunk, in insertion order."""
        if self._store is None:
            return
        for doc_id in self._store.index_to_docstore_id.values():
            document = self._store.docstore.search(doc_id)
            if isinstance(document, Document):
                yield document

    def stats(self) -> dict[str, Any]:
        """Summary for the CLI and the Phase 11 status endpoint."""
        if not self.exists and self._store is None:
            return {
                "name": self.name,
                "directory": str(self.directory),
                "built": False,
                "vector_count": 0,
            }

        self.load()
        sources = Counter(str(doc.metadata.get("source", "unknown")) for doc in self.documents())
        files = {
            str(doc.metadata.get("file_path"))
            for doc in self.documents()
            if doc.metadata.get("file_path")
        }
        return {
            "name": self.name,
            "directory": str(self.directory),
            "built": True,
            "vector_count": self.count(),
            "dimension": self.dimension(),
            "sources": dict(sorted(sources.items())),
            "document_count": len(files),
            "manifest": self.manifest(),
        }


# --- Factories ---------------------------------------------------------------


def policy_index(
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> VectorIndex:
    """The regulatory/policy index described in the architecture (Phase 4)."""
    settings = settings or get_settings()
    return VectorIndex(
        settings.policy_index_dir,
        embeddings or build_embeddings(settings),
        name="policy",
        settings=settings,
    )


def company_index(
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> VectorIndex:
    """The company/fundamentals/news index, populated in Phase 6."""
    settings = settings or get_settings()
    return VectorIndex(
        settings.company_index_dir,
        embeddings or build_embeddings(settings),
        name="company",
        settings=settings,
    )
