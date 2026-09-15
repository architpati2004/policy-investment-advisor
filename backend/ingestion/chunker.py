"""Chunking, the last step before embedding.

Splitting is separated from loading so chunk size can be retuned without
re-reading source files, and so the same splitter serves PDFs, news articles and
company filings alike.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import Settings, get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

#: Split on the largest natural boundary that fits, falling back to smaller ones.
#: Regulatory text is numbered (``3.1``, ``(a)``), so clause breaks are tried
#: before sentence breaks to avoid severing a clause from its number.
SEPARATORS: list[str] = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def build_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    """Construct a splitter from configured chunk size and overlap."""
    settings = settings or get_settings()
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({settings.chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE ({settings.chunk_size})"
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def chunk_id(text: str, metadata: dict) -> str:
    """Deterministic id for a chunk.

    Derived from content plus source and page, so re-ingesting an unchanged
    document yields identical ids. Phase 4 uses this to skip duplicates instead
    of growing the index on every run.
    """
    basis = f"{metadata.get('file_path') or metadata.get('url') or ''}|{metadata.get('page', '')}|{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks, preserving provenance.

    Every chunk inherits its parent's metadata — crucially the page number — and
    gains ``chunk_index`` (position within the parent) and ``chunk_id``.

    Args:
        documents: Per-page or per-article documents from a loader.
        settings: Overrides for chunk size and overlap.

    Returns:
        Chunked documents, in source order. Empty input returns empty output.
    """
    if not documents:
        return []

    splitter = build_splitter(settings)
    chunks: list[Document] = []

    for parent in documents:
        pieces = splitter.split_text(parent.page_content)
        for index, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            metadata = {**parent.metadata, "chunk_index": index}
            metadata["chunk_id"] = chunk_id(text, metadata)
            chunks.append(Document(page_content=text, metadata=metadata))

    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def deduplicate(chunks: list[Document]) -> list[Document]:
    """Drop chunks whose ``chunk_id`` has already been seen.

    Overlapping source documents are common — a circular reproduced inside a
    monthly bulletin, say — and duplicate chunks crowd out genuinely different
    evidence in a top-k retrieval.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        identifier = chunk.metadata.get("chunk_id")
        if identifier in seen:
            continue
        if identifier:
            seen.add(identifier)
        unique.append(chunk)

    dropped = len(chunks) - len(unique)
    if dropped:
        logger.info("Dropped %d duplicate chunks", dropped)
    return unique
