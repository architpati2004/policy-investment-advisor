"""Metadata schema attached to every chunk in either FAISS index.

A retrieved chunk is only useful if the answer can cite it, so metadata is
modelled explicitly rather than left as a loose dict. Ingestion builds these;
retrieval reads them back to produce the ``sources`` block in every RAG answer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from typing import Any, Literal

DocumentType = Literal[
    "policy",
    "circular",
    "budget",
    "regulation",
    "annual_report",
    "quarterly_result",
    "fundamentals",
    "news",
    "other",
]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class DocumentMetadata:
    """Provenance for one source document.

    Attributes:
        source: Publisher or feed name, e.g. ``"RBI"``, ``"SEBI"``, ``"Mint"``.
        document_type: One of :data:`DocumentType`.
        title: Human-readable title used in citations.
        date: Publication date as ``YYYY-MM-DD``; ``None`` when unknown.
        company: Ticker or company name, for company-index documents.
        sector: Sector label, used to match documents against holdings.
        url: Canonical link back to the original.
        file_path: Local path the document was ingested from.
    """

    source: str
    document_type: DocumentType = "other"
    title: str | None = None
    date: str | None = None
    company: str | None = None
    sector: str | None = None
    url: str | None = None
    file_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source or not self.source.strip():
            raise ValueError("metadata.source is required — citations need a publisher")
        self.source = self.source.strip()
        if self.date is not None:
            self.date = normalise_date(self.date)

    def to_dict(self) -> dict[str, Any]:
        """Flatten for storage in a LangChain ``Document.metadata``.

        FAISS metadata is flat, so ``extra`` is merged in rather than nested, and
        ``None`` values are dropped to keep the payload small.
        """
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        merged = {**extra, **{k: v for k, v in data.items() if v is not None}}
        return merged


def normalise_date(value: str | date_cls) -> str:
    """Coerce a date into ``YYYY-MM-DD``.

    Raises:
        ValueError: if the value cannot be interpreted as a calendar date.
    """
    if isinstance(value, date_cls):
        return value.isoformat()

    text = str(value).strip()
    if _ISO_DATE.match(text):
        try:
            date_cls.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid date: {text!r}") from exc
        return text

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            from datetime import datetime

            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"unrecognised date format: {value!r} (expected YYYY-MM-DD)")
