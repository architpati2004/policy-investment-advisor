"""Phase 3 tests: PDF loading, cleaning, metadata and chunking.

Every test builds its own PDF, so the suite needs no fixture files on disk and
no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.config import get_settings
from backend.exceptions import (
    EmptyDocumentError,
    EncryptedDocumentError,
    InvalidDocumentError,
)
from backend.ingestion.chunker import chunk_documents, chunk_id, deduplicate
from backend.ingestion.cleaning import clean_text, is_meaningful, strip_repeated_lines
from backend.ingestion.metadata import DocumentMetadata, normalise_date
from backend.ingestion.pdf_loader import extract_pages, iter_pdfs, load_pdf, load_pdf_directory


# --- Cleaning ----------------------------------------------------------------


def test_clean_text_rejoins_hyphenated_line_breaks() -> None:
    assert "regulation" in clean_text("The regula-\ntion applies")


def test_clean_text_unwraps_soft_breaks_but_keeps_paragraphs() -> None:
    cleaned = clean_text("first line\nsecond line\n\nnew paragraph")
    assert "first line second line" in cleaned
    assert "\n\n" in cleaned


def test_clean_text_strips_standalone_page_numbers() -> None:
    cleaned = clean_text("Body text here.\n\n12\n\nMore body text.")
    assert "\n12\n" not in cleaned
    assert "More body text." in cleaned


def test_clean_text_collapses_whitespace_and_handles_empty() -> None:
    assert clean_text("a     b") == "a b"
    assert clean_text("") == ""


def test_is_meaningful_rejects_near_empty_pages() -> None:
    assert is_meaningful("x" * 60)
    assert not is_meaningful("   ")
    assert not is_meaningful("Page 3")


def test_strip_repeated_lines_removes_running_headers() -> None:
    header = "RESERVE BANK BULLETIN"
    pages = [f"{header}\nbody {i}" for i in range(6)]
    cleaned = strip_repeated_lines(pages)
    assert all(header not in page for page in cleaned)
    assert all(f"body {i}" in cleaned[i] for i in range(6))


def test_strip_repeated_lines_leaves_short_documents_alone() -> None:
    pages = ["same\nunique one", "same\nunique two"]
    assert strip_repeated_lines(pages) == pages


# --- Metadata ----------------------------------------------------------------


def test_metadata_requires_a_source() -> None:
    with pytest.raises(ValueError):
        DocumentMetadata(source="  ")


def test_metadata_to_dict_drops_nones_and_flattens_extra() -> None:
    data = DocumentMetadata(
        source="RBI",
        document_type="circular",
        title="Sample",
        extra={"circular_no": "7/2026"},
    ).to_dict()
    assert data["source"] == "RBI"
    assert data["circular_no"] == "7/2026"
    assert "company" not in data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2026-08-10", "2026-08-10"), ("10-08-2026", "2026-08-10"), ("10 Aug 2026", "2026-08-10")],
)
def test_normalise_date_accepts_common_formats(raw: str, expected: str) -> None:
    assert normalise_date(raw) == expected


def test_normalise_date_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        normalise_date("last Tuesday")


# --- PDF loading -------------------------------------------------------------


@pytest.fixture
def policy_metadata() -> DocumentMetadata:
    return DocumentMetadata(source="RBI", document_type="circular", date="2026-08-10")


def test_load_pdf_returns_one_document_per_page(pdf_factory, policy_metadata) -> None:
    path = pdf_factory(
        "notice.pdf",
        [
            ["The repo rate remains unchanged at 6.50 per cent for the quarter ahead."],
            ["Risk weights on unsecured retail exposures rise to 125 per cent shortly."],
        ],
    )
    documents = load_pdf(path, policy_metadata)
    assert len(documents) == 2
    assert [doc.metadata["page"] for doc in documents] == [1, 2]
    assert "repo rate" in documents[0].page_content


def test_load_pdf_attaches_full_provenance(pdf_factory, policy_metadata) -> None:
    path = pdf_factory("circular.pdf", [["Text long enough to be treated as meaningful content here."]])
    metadata = load_pdf(path, policy_metadata)[0].metadata
    assert metadata["source"] == "RBI"
    assert metadata["document_type"] == "circular"
    assert metadata["date"] == "2026-08-10"
    assert metadata["title"] == "circular"
    assert metadata["file_path"].endswith("circular.pdf")


def test_load_pdf_skips_blank_pages(pdf_factory, policy_metadata) -> None:
    path = pdf_factory(
        "gappy.pdf",
        [
            ["A first page with plenty of substantive text on it to be retained."],
            [""],
            ["A third page also carrying enough text to survive the filter here."],
        ],
    )
    documents = load_pdf(path, policy_metadata)
    assert [doc.metadata["page"] for doc in documents] == [1, 3]


def test_load_pdf_rejects_missing_file(policy_metadata, tmp_path: Path) -> None:
    with pytest.raises(InvalidDocumentError):
        load_pdf(tmp_path / "nope.pdf", policy_metadata)


def test_load_pdf_rejects_non_pdf(policy_metadata, tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("plain text")
    with pytest.raises(InvalidDocumentError):
        load_pdf(other, policy_metadata)


def test_load_pdf_rejects_corrupt_file(policy_metadata, tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF at all")
    with pytest.raises(InvalidDocumentError):
        load_pdf(broken, policy_metadata)


def test_load_pdf_reports_scanned_documents(pdf_factory, policy_metadata) -> None:
    path = pdf_factory("scan.pdf", [[""], [""]])
    with pytest.raises(EmptyDocumentError) as exc:
        load_pdf(path, policy_metadata)
    assert "OCR" in str(exc.value)


def test_load_pdf_reports_encrypted_documents(pdf_factory, policy_metadata, tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    plain = pdf_factory("plain.pdf", [["Some text that will be locked away behind a password."]])
    writer = PdfWriter()
    for page in PdfReader(str(plain)).pages:
        writer.add_page(page)
    writer.encrypt("secret")
    locked = tmp_path / "locked.pdf"
    with locked.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(EncryptedDocumentError):
        load_pdf(locked, policy_metadata)


def test_extract_pages_preserves_page_positions(pdf_factory) -> None:
    path = pdf_factory("three.pdf", [["one"], [""], ["three"]])
    pages = extract_pages(path)
    assert len(pages) == 3
    assert pages[1] == ""


# --- Directory loading -------------------------------------------------------


def test_load_pdf_directory_reads_every_file(pdf_factory, tmp_path: Path) -> None:
    pdf_factory("a.pdf", [["Document A contains a sufficient quantity of readable text."]])
    pdf_factory("b.pdf", [["Document B also contains a sufficient quantity of text."]])
    documents = load_pdf_directory(tmp_path, lambda p: DocumentMetadata(source="RBI"))
    assert len(documents) == 2
    assert {Path(d.metadata["file_path"]).name for d in documents} == {"a.pdf", "b.pdf"}


def test_load_pdf_directory_skips_broken_files(pdf_factory, tmp_path: Path) -> None:
    pdf_factory("good.pdf", [["A perfectly readable document with plenty of text in it."]])
    (tmp_path / "bad.pdf").write_bytes(b"corrupt")
    documents = load_pdf_directory(tmp_path, lambda p: DocumentMetadata(source="RBI"))
    assert len(documents) == 1


def test_load_pdf_directory_can_fail_fast(pdf_factory, tmp_path: Path) -> None:
    (tmp_path / "bad.pdf").write_bytes(b"corrupt")
    with pytest.raises(InvalidDocumentError):
        load_pdf_directory(tmp_path, lambda p: DocumentMetadata(source="RBI"), skip_failures=False)


def test_iter_pdfs_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidDocumentError):
        list(iter_pdfs(tmp_path / "missing"))


# --- Chunking ----------------------------------------------------------------


def _document(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": "RBI", "page": page, "file_path": "/x.pdf"})


def test_chunking_respects_configured_size() -> None:
    settings = get_settings()
    chunks = chunk_documents([_document("sentence. " * 600)], settings)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= settings.chunk_size for chunk in chunks)


def test_chunks_inherit_page_metadata() -> None:
    chunks = chunk_documents([_document("word " * 500, page=7)])
    assert chunks
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_short_document_stays_one_chunk() -> None:
    chunks = chunk_documents([_document("A short policy paragraph.")])
    assert len(chunks) == 1
    assert chunks[0].page_content == "A short policy paragraph."


def test_chunking_empty_input_returns_empty() -> None:
    assert chunk_documents([]) == []


def test_chunk_ids_are_stable_across_runs() -> None:
    document = _document("Identical text for both ingestion runs of this document.")
    first = chunk_documents([document])
    second = chunk_documents([document])
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]


def test_chunk_id_differs_by_page() -> None:
    text = "The same sentence appearing on two different pages."
    assert chunk_id(text, {"file_path": "/x.pdf", "page": 1}) != chunk_id(
        text, {"file_path": "/x.pdf", "page": 2}
    )


def test_deduplicate_drops_repeats() -> None:
    chunks = chunk_documents([_document("Repeated content."), _document("Repeated content.")])
    assert len(chunks) == 2
    assert len(deduplicate(chunks)) == 1


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings().model_copy(update={"chunk_size": 100, "chunk_overlap": 100})
    with pytest.raises(ValueError):
        chunk_documents([_document("text")], settings)


# --- End to end --------------------------------------------------------------


def test_pdf_to_chunks_preserves_citation_fields(pdf_factory) -> None:
    """A chunk must carry everything a citation needs: source, date, page."""
    path = pdf_factory(
        "policy.pdf",
        [["Paragraph one of the notice. " * 40], ["Paragraph two of the notice. " * 40]],
    )
    metadata = DocumentMetadata(source="SEBI", document_type="regulation", date="2026-03-01")
    chunks = deduplicate(chunk_documents(load_pdf(path, metadata)))

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "SEBI"
        assert chunk.metadata["date"] == "2026-03-01"
        assert chunk.metadata["page"] in (1, 2)
        assert chunk.metadata["chunk_id"]
