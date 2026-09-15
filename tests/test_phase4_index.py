"""Phase 4 tests: embeddings and the policy FAISS index.

The unit tests use a deterministic in-process embedding model, so the whole
suite runs with Ollama stopped and still exercises real FAISS indexes on real
temporary directories. The integration tests at the bottom talk to the local
embedding model and skip themselves when it is not available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import Settings, get_settings
from backend.exceptions import (
    EmbeddingError,
    IndexCorruptError,
    IndexModelMismatchError,
    IndexNotFoundError,
    LLMTimeoutError,
    ModelNotInstalledError,
    OllamaUnavailableError,
)
from backend.ingestion.chunker import chunk_documents, chunk_id
from backend.rag import embeddings as embeddings_module
from backend.rag import ollama_client
from backend.rag.embeddings import LocalEmbeddings, build_embeddings
from backend.rag.vector_store import (
    INDEX_NAME,
    SearchResult,
    VectorIndex,
    company_index,
    policy_index,
)

WORD = re.compile(r"[a-z0-9]+")


# --- Test doubles ------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings, unit length, no network.

    Hashing each word into a fixed number of buckets means texts sharing
    vocabulary end up close together, so retrieval *ordering* can be asserted
    without a real model — while staying reproducible across runs.
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


class StubClient:
    """Stands in for ``OllamaEmbeddings``, recording the batches it receives."""

    def __init__(self, vector: list[float] | None = None, error: Exception | None = None) -> None:
        self.vector = vector if vector is not None else [3.0, 4.0]
        self.error = error
        self.batches: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.error:
            raise self.error
        self.batches.append(list(texts))
        return [list(self.vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.error:
            raise self.error
        self.query_calls.append(text)
        return list(self.vector)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary vector store, never the project's own."""
    return get_settings().model_copy(update={"vectorstore_dir": str(tmp_path / "vectorstore")})


@pytest.fixture
def index(tmp_path: Path, settings: Settings) -> VectorIndex:
    return VectorIndex(
        tmp_path / "vectorstore" / "policy_index",
        FakeEmbeddings(),
        name="policy",
        settings=settings,
    )


def _chunks(*texts: str, source: str = "RBI", page: int = 1) -> list[Document]:
    """Chunk-shaped documents with the metadata ingestion would attach."""
    documents = []
    for position, text in enumerate(texts):
        metadata: dict[str, Any] = {
            "source": source,
            "document_type": "circular",
            "title": "Master Circular",
            "date": "2026-08-10",
            "file_path": f"/data/policies/{source.lower()}.pdf",
            "page": page + position,
        }
        metadata["chunk_id"] = chunk_id(text, metadata)
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


REPO = "The repo rate remains unchanged at 6.50 per cent after the policy review."
RISK = "Risk weights on unsecured retail credit exposures rise to 125 per cent."
MONSOON = "Monsoon rainfall was above the long period average across the peninsula."


# --- Embeddings: batching ----------------------------------------------------


def _embeddings(client: StubClient, *, batch_size: int = 2) -> LocalEmbeddings:
    return LocalEmbeddings(
        model="test-embed",
        base_url="http://localhost:11434",
        batch_size=batch_size,
        timeout_seconds=30,
        settings=get_settings(),
        client=client,
    )


def test_embedding_batches_never_exceed_the_configured_size() -> None:
    client = StubClient()
    _embeddings(client, batch_size=2).embed_documents([f"text {i}" for i in range(5)])
    assert [len(batch) for batch in client.batches] == [2, 2, 1]


def test_embedding_returns_one_vector_per_text_in_order() -> None:
    client = StubClient()
    texts = [f"text {i}" for i in range(5)]
    vectors = _embeddings(client, batch_size=2).embed_documents(texts)
    assert len(vectors) == len(texts)
    assert [text for batch in client.batches for text in batch] == texts


def test_embedding_empty_input_makes_no_calls() -> None:
    client = StubClient()
    assert _embeddings(client).embed_documents([]) == []
    assert client.batches == []


def test_vectors_are_normalised_to_unit_length() -> None:
    """Cosine scoring in the index depends on this; 3-4-5 makes it checkable."""
    vectors = _embeddings(StubClient([3.0, 4.0])).embed_documents(["anything"])
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(v * v for v in vectors[0])) == pytest.approx(1.0)


def test_query_vectors_are_normalised_too() -> None:
    vector = _embeddings(StubClient([0.0, 5.0])).embed_query("a question")
    assert vector == pytest.approx([0.0, 1.0])


def test_zero_vector_does_not_divide_by_zero() -> None:
    assert _embeddings(StubClient([0.0, 0.0])).embed_documents(["  "])[0] == [0.0, 0.0]


def test_dimension_is_probed_once_and_cached() -> None:
    client = StubClient([1.0] * 7)
    embeddings = _embeddings(client)
    assert embeddings.dimension == 7
    assert embeddings.dimension == 7
    assert len(client.query_calls) == 1


def test_dimension_is_learned_from_documents_without_an_extra_call() -> None:
    client = StubClient([1.0] * 7)
    embeddings = _embeddings(client)
    embeddings.embed_documents(["text"])
    assert embeddings.dimension == 7
    assert client.query_calls == []


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _embeddings(StubClient(), batch_size=0)


def test_short_batch_response_is_reported_not_silently_misaligned() -> None:
    """A vector list shorter than its batch would shift every later chunk's id."""

    class ShortClient(StubClient):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    with pytest.raises(EmbeddingError):
        _embeddings(ShortClient(), batch_size=4).embed_documents(["a", "b", "c"])


# --- Embeddings: error translation -------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (ConnectionError("connection refused"), OllamaUnavailableError),
        (RuntimeError('model "embeddinggemma" not found, try pulling it first'), ModelNotInstalledError),
        (TimeoutError("read timed out"), LLMTimeoutError),
        (RuntimeError("something else went wrong"), EmbeddingError),
    ],
)
def test_client_errors_become_named_application_errors(raised: Exception, expected: type) -> None:
    with pytest.raises(expected):
        _embeddings(StubClient(error=raised)).embed_documents(["text"])


def test_query_errors_are_translated_as_well() -> None:
    error = RuntimeError('model "embeddinggemma" not found, try pulling it first')
    with pytest.raises(ModelNotInstalledError) as exc:
        _embeddings(StubClient(error=error)).embed_query("text")
    assert "ollama pull" in str(exc.value)


def test_build_embeddings_reads_configuration() -> None:
    configured = get_settings().model_copy(
        update={"embedding_model": "some-model", "embedding_batch_size": 3}
    )
    embeddings = build_embeddings(configured, client=StubClient())
    assert embeddings.model == "some-model"
    assert embeddings.batch_size == 3


def test_verify_ready_surfaces_a_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(model: str, settings: Settings | None = None) -> None:
        raise ModelNotInstalledError(model, ["qwen3:4b"])

    monkeypatch.setattr(ollama_client, "require_model", _missing)
    with pytest.raises(ModelNotInstalledError):
        embeddings_module.verify_ready(get_settings())


# --- Index: building, searching, metadata ------------------------------------


def test_search_returns_scored_results_with_metadata_intact(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, RISK))
    results = index.search("repo rate", min_score=0.0)

    assert results and isinstance(results[0], SearchResult)
    top = results[0]
    assert top.metadata["source"] == "RBI"
    assert top.metadata["date"] == "2026-08-10"
    assert top.metadata["page"] == 1
    assert top.metadata["chunk_id"]
    assert 0.0 <= top.score <= 1.0


def test_results_are_ordered_by_relevance(index: VectorIndex) -> None:
    index.add_documents(_chunks(MONSOON, RISK, REPO))
    results = index.search("what happened to the repo rate", min_score=0.0)

    assert "repo rate" in results[0].text
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_min_score_drops_weak_matches(index: VectorIndex) -> None:
    index.add_documents(_chunks(MONSOON))
    assert index.search("repo rate policy decision", min_score=0.9) == []
    assert index.search("monsoon rainfall", min_score=0.0)


def test_k_limits_the_number_of_results(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, RISK, MONSOON))
    assert len(index.search("policy", k=2, min_score=0.0)) <= 2
    assert index.search("policy", k=0, min_score=0.0) == []


def test_metadata_filter_restricts_the_search(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, source="RBI"))
    index.add_documents(_chunks(RISK, source="SEBI"))

    results = index.search("policy", filter={"source": "SEBI"}, min_score=0.0)
    assert results
    assert {r.metadata["source"] for r in results} == {"SEBI"}


def test_citation_reads_like_a_citation(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO))
    citation = index.search("repo rate", min_score=0.0)[0].citation()
    assert "RBI" in citation and "p. 1" in citation and "2026-08-10" in citation


def test_result_serialises_for_the_api(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO))
    payload = index.search("repo rate", min_score=0.0)[0].to_dict()
    assert set(payload) == {"text", "score", "citation", "metadata"}
    assert json.dumps(payload)  # must survive a JSON response


# --- Index: persistence ------------------------------------------------------


def test_index_round_trips_through_disk(tmp_path: Path, settings: Settings) -> None:
    directory = tmp_path / "vectorstore" / "policy_index"
    original = VectorIndex(directory, FakeEmbeddings(), name="policy", settings=settings)
    original.add_documents(_chunks(REPO, RISK, MONSOON))

    assert (directory / f"{INDEX_NAME}.faiss").is_file()
    assert (directory / f"{INDEX_NAME}.pkl").is_file()

    reloaded = VectorIndex(directory, FakeEmbeddings(), name="policy", settings=settings)
    assert reloaded.exists
    reloaded.load()

    assert reloaded.count() == original.count() == 3
    before = original.search("repo rate", min_score=0.0)
    after = reloaded.search("repo rate", min_score=0.0)
    assert [r.text for r in after] == [r.text for r in before]
    assert [r.score for r in after] == pytest.approx([r.score for r in before])
    assert after[0].metadata["chunk_id"] == before[0].metadata["chunk_id"]


def test_manifest_records_the_model_and_dimension(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO))
    manifest = index.manifest()
    assert manifest["embedding_model"] == get_settings().embedding_model
    assert manifest["dimension"] == FakeEmbeddings.dimension
    assert manifest["vector_count"] == 1
    assert manifest["distance"] == "cosine"
    assert manifest["updated_at"]


def test_loading_an_index_built_by_another_model_is_refused(
    tmp_path: Path, settings: Settings
) -> None:
    """Vectors from two models share no geometry — loading would return nonsense."""
    directory = tmp_path / "vectorstore" / "policy_index"
    VectorIndex(directory, FakeEmbeddings(), name="policy", settings=settings).add_documents(
        _chunks(REPO)
    )

    switched = settings.model_copy(update={"embedding_model": "a-different-model"})
    index = VectorIndex(directory, FakeEmbeddings(), name="policy", settings=switched)
    with pytest.raises(IndexModelMismatchError) as exc:
        index.load()
    assert "--rebuild" in str(exc.value)


def test_searching_before_building_says_so(index: VectorIndex) -> None:
    with pytest.raises(IndexNotFoundError) as exc:
        index.search("anything")
    assert "build_policy_index.py" in str(exc.value)


def test_a_damaged_index_reports_clearly(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO))
    index.index_file.write_bytes(b"not a faiss index")

    fresh = VectorIndex(index.directory, FakeEmbeddings(), name="policy", settings=index._settings)
    with pytest.raises(IndexCorruptError):
        fresh.load()


def test_clear_removes_every_file(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO))
    index.clear()
    assert not index.exists
    assert not index.manifest_file.exists()
    assert index.chunk_ids() == set()


# --- Index: deduplication ----------------------------------------------------


def test_reingesting_the_same_document_adds_nothing(index: VectorIndex) -> None:
    chunks = _chunks(REPO, RISK)
    first = index.add_documents(chunks)
    second = index.add_documents(chunks)

    assert (first.added, first.skipped) == (2, 0)
    assert (second.added, second.skipped) == (0, 2)
    assert index.count() == 2


def test_reingest_adds_only_the_new_chunks(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, RISK))
    report = index.add_documents(_chunks(REPO, RISK) + _chunks(MONSOON, page=9))

    assert (report.added, report.skipped) == (1, 2)
    assert index.count() == 3


def test_deduplication_survives_a_reload(tmp_path: Path, settings: Settings) -> None:
    """The skip must be decided from disk, not from an in-memory set."""
    directory = tmp_path / "vectorstore" / "policy_index"
    VectorIndex(directory, FakeEmbeddings(), name="policy", settings=settings).add_documents(
        _chunks(REPO)
    )

    reopened = VectorIndex(directory, FakeEmbeddings(), name="policy", settings=settings)
    report = reopened.add_documents(_chunks(REPO))
    assert (report.added, report.skipped) == (0, 1)
    assert reopened.count() == 1


def test_duplicates_inside_one_batch_are_collapsed(index: VectorIndex) -> None:
    """FAISS rejects a batch carrying the same id twice, so we must filter first."""
    report = index.add_documents(_chunks(REPO) + _chunks(REPO))
    assert (report.added, report.skipped) == (1, 1)
    assert index.count() == 1


def test_chunks_without_an_id_still_deduplicate(index: VectorIndex) -> None:
    bare = Document(page_content=REPO, metadata={"source": "RBI", "page": 1})
    index.add_documents([bare])
    report = index.add_documents([Document(page_content=REPO, metadata={"source": "RBI", "page": 1})])
    assert (report.added, report.skipped) == (0, 1)


def test_adding_nothing_builds_nothing(index: VectorIndex) -> None:
    report = index.add_documents([])
    assert (report.added, report.skipped, report.total) == (0, 0, 0)
    assert not index.exists


# --- Index: introspection and factories --------------------------------------


def test_stats_before_and_after_building(index: VectorIndex) -> None:
    empty = index.stats()
    assert empty["built"] is False and empty["vector_count"] == 0

    index.add_documents(_chunks(REPO, source="RBI") + _chunks(RISK, source="SEBI", page=4))
    summary = index.stats()
    assert summary["built"] is True
    assert summary["vector_count"] == 2
    assert summary["dimension"] == FakeEmbeddings.dimension
    assert summary["sources"] == {"RBI": 1, "SEBI": 1}
    assert summary["document_count"] == 2


def test_documents_iterates_everything_indexed(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, RISK))
    texts = [document.page_content for document in index.documents()]
    assert sorted(texts) == sorted([REPO, RISK])


def test_retriever_is_available_for_the_phase_5_chain(index: VectorIndex) -> None:
    index.add_documents(_chunks(REPO, RISK, MONSOON))
    documents = index.as_retriever(search_kwargs={"k": 2}).invoke("repo rate")
    assert len(documents) == 2
    assert "repo rate" in documents[0].page_content


def test_factories_point_at_the_configured_directories(settings: Settings) -> None:
    policy = policy_index(FakeEmbeddings(), settings)
    company = company_index(FakeEmbeddings(), settings)
    assert policy.directory == settings.policy_index_dir
    assert company.directory == settings.company_index_dir
    assert policy.name == "policy" and company.name == "company"


def test_pdf_chunks_index_and_retrieve_end_to_end(
    pdf_factory, tmp_path: Path, settings: Settings
) -> None:
    """Phase 3 output feeds Phase 4 input without translation in between."""
    from backend.ingestion.metadata import DocumentMetadata
    from backend.ingestion.pdf_loader import load_pdf

    path = pdf_factory(
        "circular.pdf",
        [[f"{REPO} " * 20], [f"{RISK} " * 20]],
    )
    pages = load_pdf(path, DocumentMetadata(source="RBI", document_type="circular", date="2026-08-10"))
    chunks = chunk_documents(pages, settings)

    index = VectorIndex(
        tmp_path / "vectorstore" / "policy_index",
        FakeEmbeddings(),
        name="policy",
        settings=settings,
    )
    report = index.add_documents(chunks)
    assert report.added == len(chunks)

    results = index.search("risk weights on unsecured retail credit", min_score=0.0)
    assert results
    assert "Risk weights" in results[0].text
    assert results[0].metadata["page"] == 2
    assert results[0].metadata["file_path"].endswith("circular.pdf")


# --- Integration (skipped unless the embedding model is available) -----------


def _embeddings_available() -> bool:
    try:
        return ollama_client.check_status().embedding_available
    except Exception:  # noqa: BLE001
        return False


requires_embeddings = pytest.mark.skipif(
    not _embeddings_available(), reason="Ollama embedding model is not available"
)


@pytest.mark.integration
@requires_embeddings
def test_live_embeddings_have_a_stable_dimension() -> None:
    embeddings = build_embeddings()
    vectors = embeddings.embed_documents([REPO, MONSOON])

    assert len({len(vector) for vector in vectors}) == 1
    assert len(vectors[0]) == embeddings.dimension > 0
    assert math.sqrt(sum(v * v for v in vectors[0])) == pytest.approx(1.0, abs=1e-6)
    assert len(embeddings.embed_query("repo rate")) == embeddings.dimension


@pytest.mark.integration
@requires_embeddings
def test_live_index_round_trip_and_ranking(tmp_path: Path, settings: Settings) -> None:
    directory = tmp_path / "vectorstore" / "policy_index"
    index = VectorIndex(directory, build_embeddings(settings), name="policy", settings=settings)
    index.add_documents(_chunks(REPO, RISK, MONSOON))

    reloaded = VectorIndex(directory, build_embeddings(settings), name="policy", settings=settings)
    results = reloaded.search("what did the central bank decide about the repo rate?", min_score=0.0)

    assert reloaded.count() == 3
    assert "repo rate" in results[0].text
    assert results[0].score > results[-1].score
