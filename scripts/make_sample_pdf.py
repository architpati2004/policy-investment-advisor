"""Generate a synthetic sample PDF for testing ingestion.

Real RBI and SEBI circulars are the eventual input, but you need *something* to
ingest before you have downloaded any. This writes a clearly-labelled synthetic
document into ``data/policies/``.

The sample is marked SYNTHETIC on every page on purpose. This system answers
questions about real financial regulation, and a realistic-looking fake circular
sitting in the index would eventually be retrieved and cited as though it were
genuine. Delete it once you have real documents.

Run with::

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _escape(text: str) -> str:
    """Escape characters that terminate or nest inside a PDF string literal."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    """Build a page content stream drawing each line of text."""
    body = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        body.append(f"({_escape(line)}) Tj")
        body.append("T*")
    body.append("ET")
    return "\n".join(body).encode("latin-1")


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a multi-page PDF containing the given lines of text.

    A minimal but standards-valid PDF writer, so neither the tests nor this
    script need a PDF-generation dependency.

    Args:
        path: Destination file.
        pages: One list of text lines per page.

    Returns:
        The path written.
    """
    objects: list[bytes] = []
    page_count = len(pages)
    page_obj_ids = [4 + i * 2 for i in range(page_count)]
    kids = " ".join(f"{oid} 0 R" for oid in page_obj_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, lines in enumerate(pages):
        stream = _content_stream(lines)
        content_id = page_obj_ids[index] + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


BANNER = "SYNTHETIC SAMPLE DOCUMENT - NOT A REAL REGULATORY PUBLICATION"

SAMPLE_PAGES: list[list[str]] = [
    [
        BANNER,
        "Fictional Central Authority - Sample Policy Notice",
        "Notice No. SAMPLE/2026/01",
        "",
        "1. Introduction",
        "This document exists only to exercise the ingestion pipeline of the",
        "Policy-Based Investment Advisor. Every figure, rule and institution named",
        "below is invented. Nothing here describes real regulation and nothing here",
        "should be relied upon for any purpose.",
        "",
        "2. Illustrative Rate Decision",
        "The Committee has resolved to hold the benchmark policy rate at 6.50 per",
        "cent. Members noted that headline inflation has moderated while credit",
        "growth in the retail segment remains elevated.",
    ],
    [
        BANNER,
        "3. Illustrative Provisions for Lending Institutions",
        "",
        "3.1 Covered institutions shall maintain a liquidity coverage ratio of not",
        "less than 110 per cent, measured on a fortnightly average basis.",
        "",
        "3.2 Unsecured retail exposures shall carry a risk weight of 125 per cent,",
        "raised from 100 per cent, with effect from the beginning of the next",
        "financial quarter.",
        "",
        "3.3 Institutions shall report compliance with paragraph 3.2 within thirty",
        "days of the effective date.",
    ],
    [
        BANNER,
        "4. Illustrative Market Measures",
        "",
        "4.1 Settlement for listed equity shall follow a same-day cycle for the",
        "scrips specified in the annexure, applied in phases.",
        "",
        "4.2 Asset management companies shall disclose portfolio concentration",
        "above ten per cent of net assets in a single issuer.",
        "",
        "5. Sectors Referenced",
        "Banking and financial services, capital markets, energy.",
        "",
        "End of synthetic sample document.",
    ],
]


def main() -> int:
    from backend.config import get_settings

    settings = get_settings()
    settings.ensure_directories()
    target = settings.policy_data_dir / "SAMPLE_synthetic_policy_notice.pdf"
    write_pdf(target, SAMPLE_PAGES)
    sys.stdout.write(f"Wrote {target} ({len(SAMPLE_PAGES)} pages)\n")
    sys.stdout.write("This is synthetic test data. Delete it once you have real documents.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
