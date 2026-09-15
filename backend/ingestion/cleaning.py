"""Text cleaning for extracted document text.

PDF extraction produces text that is technically correct and semantically awful:
words split across line breaks, running headers repeated on every page, ragged
whitespace. None of that survives embedding well — a chunk containing
``"repo rate re-\nmains"`` will not match a query about the repo rate as closely
as it should. Cleaning happens once, before chunking.
"""

from __future__ import annotations

import re
import unicodedata

# Hyphen at end of line joining a split word: "regula-\ntion" -> "regulation".
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# A single newline inside a sentence is a wrap, not a paragraph break.
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
# Standalone page markers: "12", "Page 12", "- 12 -".
_PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE | re.MULTILINE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Below this, a page is almost certainly blank, a cover, or a scan.
MIN_MEANINGFUL_CHARS = 50


def clean_text(text: str) -> str:
    """Normalise extracted text without changing its meaning.

    Joins hyphenated line breaks, unwraps soft line breaks while preserving
    paragraph boundaries, strips page-number artefacts and collapses whitespace.
    """
    if not text:
        return ""

    # NFKC folds ligatures and full-width forms into plain ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _PAGE_MARKER.sub("", text)
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)

    return "\n".join(line.strip() for line in text.split("\n")).strip()


def is_meaningful(text: str, min_chars: int = MIN_MEANINGFUL_CHARS) -> bool:
    """True when a page carries enough text to be worth embedding.

    Guards against cover pages, blank pages and image-only scans, which produce
    near-empty strings that would otherwise become useless index entries.
    """
    return len(text.strip()) >= min_chars


def strip_repeated_lines(pages: list[str], threshold: float = 0.6) -> list[str]:
    """Remove running headers and footers shared across most pages.

    A line appearing on more than ``threshold`` of pages is boilerplate — a
    document title banner or footer. Left in, it dominates short chunks and
    pollutes similarity scores. Skipped for documents under four pages, where
    repetition is more likely to be genuine content.
    """
    if len(pages) < 4:
        return pages

    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.split("\n") if ln.strip()}:
            counts[line] += 1

    cutoff = max(2, int(len(pages) * threshold))
    boilerplate = {line for line, count in counts.items() if count >= cutoff and len(line) < 120}
    if not boilerplate:
        return pages

    cleaned = []
    for page in pages:
        kept = [ln for ln in page.split("\n") if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept).strip())
    return cleaned
