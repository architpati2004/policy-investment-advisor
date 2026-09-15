"""Phase 3 checker: preview what ingestion does to a PDF.

Shows page extraction, cleaning and chunking without embedding anything, so you
can sanity-check chunk boundaries before spending time building an index.

Run with::

    python scripts/ingest_pdf.py data/policies/SAMPLE_synthetic_policy_notice.pdf
    python scripts/ingest_pdf.py data/policies/circular.pdf --source RBI --type circular
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.exceptions import AdvisorError  # noqa: E402
from backend.ingestion.chunker import chunk_documents, deduplicate  # noqa: E402
from backend.ingestion.metadata import DocumentMetadata  # noqa: E402
from backend.ingestion.pdf_loader import load_pdf  # noqa: E402

PASS, FAIL, INFO = "[ OK ]", "[FAIL]", "[INFO]"
PREVIEW_CHARS = 220


def _report(status: str, message: str) -> None:
    sys.stdout.write(f"{status} {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview PDF ingestion.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument("--source", default="SAMPLE", help="Publisher, e.g. RBI or SEBI")
    parser.add_argument("--type", dest="document_type", default="policy", help="Document type")
    parser.add_argument("--date", default=None, help="Publication date (YYYY-MM-DD)")
    parser.add_argument("--url", default=None, help="Canonical URL of the original")
    parser.add_argument("--show", type=int, default=3, help="Number of chunks to preview")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    sys.stdout.write("Phase 3 ingestion preview\n" + "-" * 60 + "\n")
    _report(INFO, f"File: {args.pdf}")
    _report(INFO, f"Chunk size {settings.chunk_size}, overlap {settings.chunk_overlap}")

    try:
        metadata = DocumentMetadata(
            source=args.source,
            document_type=args.document_type,
            date=args.date,
            url=args.url,
        )
        pages = load_pdf(args.pdf, metadata)
    except AdvisorError as exc:
        _report(FAIL, str(exc))
        return 1
    except ValueError as exc:
        _report(FAIL, f"Bad metadata: {exc}")
        return 1

    characters = sum(len(page.page_content) for page in pages)
    _report(PASS, f"Extracted {len(pages)} pages, {characters:,} characters of text")

    chunks = deduplicate(chunk_documents(pages, settings))
    if not chunks:
        _report(FAIL, "Chunking produced nothing — the document may be near-empty")
        return 1

    sizes = [len(chunk.page_content) for chunk in chunks]
    _report(PASS, f"Produced {len(chunks)} chunks")
    _report(INFO, f"  size: min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")

    pages_covered = sorted({chunk.metadata.get("page") for chunk in chunks})
    _report(PASS, f"Page numbers preserved on every chunk: {pages_covered}")

    for chunk in chunks[: max(0, args.show)]:
        meta = chunk.metadata
        sys.stdout.write("\n" + "-" * 60 + "\n")
        sys.stdout.write(
            f"chunk {meta['chunk_id']} | page {meta.get('page')} | "
            f"index {meta.get('chunk_index')} | {len(chunk.page_content)} chars\n"
        )
        preview = " ".join(chunk.page_content.split())[:PREVIEW_CHARS]
        sys.stdout.write(f"{preview}...\n")

    sys.stdout.write("\n" + "-" * 60 + "\n")
    sys.stdout.write("Ingestion works. Next: Phase 4 (embeddings and the policy FAISS index).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
