"""Endpoints describing what has been indexed, and searching it directly.

Search here is retrieval without generation: it returns the chunks and their
scores, which is what makes an answer auditable and what makes a thin answer
diagnosable. It is also fast, where anything touching the model is not.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api import schemas
from backend.api.deps import get_indexes
from backend.exceptions import IndexNotFoundError
from backend.rag.vector_store import VectorIndex

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _describe(index: VectorIndex) -> schemas.IndexOut:
    stats = index.stats()
    manifest = stats.get("manifest") or {}
    return schemas.IndexOut(
        name=index.name,
        built=bool(stats.get("built")),
        vector_count=int(stats.get("vector_count") or 0),
        dimension=stats.get("dimension"),
        directory=stats.get("directory"),
        embedding_model=manifest.get("embedding_model"),
        updated_at=manifest.get("updated_at"),
    )


@router.get("/indexes", response_model=list[schemas.IndexOut])
def read_indexes(
    indexes: dict[str, VectorIndex] = Depends(get_indexes),
) -> list[schemas.IndexOut]:
    """What each index holds. An unbuilt index reports ``built: false``, not an error."""
    return [_describe(index) for index in indexes.values()]


@router.get("/search", response_model=schemas.SearchOut)
def search(
    q: str = Query(..., min_length=1, max_length=500),
    index: str = Query("policy", pattern="^(policy|company|news)$"),
    k: int | None = Query(None, ge=1, le=50),
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    company: str | None = None,
    sector: str | None = None,
    indexes: dict[str, VectorIndex] = Depends(get_indexes),
) -> schemas.SearchOut:
    """Retrieve chunks without generating an answer.

    Raises:
        IndexNotFoundError: that index has not been built (404 with the command
            that builds it).
    """
    store = indexes[index]
    if not store.exists:
        raise IndexNotFoundError(index, str(store.directory))

    from backend.rag.retrieval import scope_filter

    results = store.search(
        q,
        k=k,
        min_score=0.0 if min_score is None else min_score,
        filter=scope_filter([company] if company else None, [sector] if sector else None),
    )
    return schemas.SearchOut(
        query=q,
        index=index,
        considered=len(results),
        results=[schemas.SearchHitOut(**hit.to_dict()) for hit in results],
        reason=None if results else "nothing in this index matched",
    )
