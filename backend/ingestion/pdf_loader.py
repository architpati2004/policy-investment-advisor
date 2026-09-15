"""PDF ingestion.

Loads a PDF into one LangChain ``Document`` per page. Page-level granularity is
deliberate: the brief requires page numbers in citations, and page identity is
lost the moment you concatenate a document into a single string.

Chunking happens later, in :mod:`backend.ingestion.chunker`, so that loading
stays independent of retrieval parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.exceptions import (
    EmptyDocumentError,
    EncryptedDocumentError,
    InvalidDocumentError,
)
from backend.ingestion.cleaning import clean_text, is_meaningful, strip_repeated_lines
from backend.ingestion.metadata import DocumentMetadata
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _open_reader(path: Path) -> PdfReader:
    """Open a PDF, converting library errors into application errors."""
    if not path.is_file():
        raise InvalidDocumentError(str(path), "file does not exist")
    if path.suffix.lower() != ".pdf":
        raise InvalidDocumentError(str(path), f"not a PDF (suffix {path.suffix!r})")

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise InvalidDocumentError(str(path), f"unreadable PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf raises assorted types on corruption
        raise InvalidDocumentError(str(path), str(exc)) from exc

    if reader.is_encrypted:
        # An empty password unlocks many "protected" government PDFs.
        try:
            if reader.decrypt("") == 0:
                raise EncryptedDocumentError(str(path))
        except EncryptedDocumentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EncryptedDocumentError(str(path)) from exc

    return reader


def extract_pages(path: Path) -> list[str]:
    """Return cleaned text for every page, preserving page order and indices.

    Pages that fail to extract become empty strings rather than disappearing, so
    a page's position in the list always matches its real page number.
    """
    reader = _open_reader(path)
    if len(reader.pages) == 0:
        raise EmptyDocumentError(str(path), "contains no pages")

    raw: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page should not kill the document
            logger.warning("Page %d of %s failed to extract: %s", number, path.name, exc)
            raw.append("")

    return [clean_text(page) for page in strip_repeated_lines(raw)]


def load_pdf(
    path: str | Path,
    metadata: DocumentMetadata,
    *,
    skip_empty_pages: bool = True,
) -> list[Document]:
    """Load one PDF into per-page documents.

    Args:
        path: Location of the PDF.
        metadata: Provenance applied to every page, with ``page`` added per page.
        skip_empty_pages: Drop pages with too little text to be worth embedding.

    Returns:
        One ``Document`` per retained page, in page order.

    Raises:
        InvalidDocumentError: missing, non-PDF, or corrupt file.
        EncryptedDocumentError: password-protected file.
        EmptyDocumentError: no pages, or no extractable text at all.
    """
    path = Path(path)
    pages = extract_pages(path)

    base = metadata.to_dict()
    base.setdefault("title", path.stem)
    base["file_path"] = str(path)

    documents: list[Document] = []
    for number, text in enumerate(pages, start=1):
        if skip_empty_pages and not is_meaningful(text):
            continue
        documents.append(Document(page_content=text, metadata={**base, "page": number}))

    if not documents:
        raise EmptyDocumentError(
            str(path),
            "no extractable text — the file is probably a scan and needs OCR",
        )

    logger.info("Loaded %s: %d/%d pages retained", path.name, len(documents), len(pages))
    return documents


def iter_pdfs(directory: str | Path) -> Iterator[Path]:
    """Yield PDF paths under a directory, sorted, recursing into subfolders."""
    directory = Path(directory)
    if not directory.is_dir():
        raise InvalidDocumentError(str(directory), "not a directory")
    yield from sorted(p for p in directory.rglob("*.pdf") if p.is_file() and not p.name.startswith("."))


def load_pdf_directory(
    directory: str | Path,
    metadata_factory,
    *,
    skip_failures: bool = True,
) -> list[Document]:
    """Load every PDF in a directory.

    Args:
        directory: Folder to scan, recursively.
        metadata_factory: Callable taking a ``Path`` and returning
            :class:`DocumentMetadata`, so per-file provenance can differ.
        skip_failures: Log and continue past unreadable files rather than
            aborting the batch. One corrupt download should not stop an ingest.

    Returns:
        Documents from every file that loaded, flattened into one list.
    """
    documents: list[Document] = []
    for pdf in iter_pdfs(directory):
        try:
            documents.extend(load_pdf(pdf, metadata_factory(pdf)))
        except Exception as exc:  # noqa: BLE001 - reported per file below
            if not skip_failures:
                raise
            logger.error("Skipped %s: %s", pdf.name, exc)
    return documents
