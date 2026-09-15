"""Loading company filings into the company index.

Thin by design: PDF reading, cleaning and chunking already exist and are not
company-specific, so this module only does the part that is — deciding which
company a file belongs to, and refusing to index it when it cannot tell.

That refusal is the point. A chunk in the company index with no ``company``
attached is worse than a missing chunk: no holding can match it, so it can never
be retrieved on purpose, but it can still surface in an unfiltered search and be
cited as though somebody had vouched for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from backend.exceptions import AdvisorError, UnknownCompanyError
from backend.ingestion.company_registry import Company, CompanyRegistry
from backend.ingestion.metadata import DocumentMetadata, DocumentType
from backend.ingestion.pdf_loader import iter_pdfs, load_pdf
from backend.logging_config import get_logger

logger = get_logger(__name__)

#: Company filings default to this when the caller does not say otherwise.
DEFAULT_DOCUMENT_TYPE: DocumentType = "annual_report"


@dataclass(frozen=True)
class SkippedFile:
    """A file that could not be ingested, and why — reported, never silent."""

    path: Path
    reason: str


def company_metadata(
    company: Company,
    *,
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE,
    date: str | None = None,
    url: str | None = None,
) -> DocumentMetadata:
    """Provenance for a filing.

    ``source`` is the company's readable name, because that is what a citation
    should show; ``company`` is the ticker, because that is what a portfolio
    holding is keyed by and the one spelling that does not drift.
    """
    return DocumentMetadata(
        source=company.name,
        document_type=document_type,
        company=company.ticker,
        sector=company.sector,
        date=date,
        url=url,
    )


def load_company_pdf(
    path: str | Path,
    company: Company,
    *,
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE,
    date: str | None = None,
) -> list[Document]:
    """Load one filing into per-page documents carrying company identity."""
    return load_pdf(path, company_metadata(company, document_type=document_type, date=date))


def resolve_company(
    path: Path,
    registry: CompanyRegistry,
    override: Company | None = None,
) -> Company:
    """Decide which company a file belongs to.

    Raises:
        UnknownCompanyError: the filename matches nothing in the registry and
            no override was supplied.
    """
    if override is not None:
        return override
    company = registry.resolve(path.name)
    if company is None:
        raise UnknownCompanyError(str(path))
    return company


def load_company_directory(
    directory: str | Path,
    registry: CompanyRegistry,
    *,
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE,
    override: Company | None = None,
    skip_failures: bool = True,
) -> tuple[list[Document], list[SkippedFile]]:
    """Load every filing in a directory that can be attributed to a company.

    Args:
        directory: Folder of company PDFs, scanned recursively.
        registry: Declared companies, used to resolve filenames.
        document_type: What kind of filing these are.
        override: Attribute every file to this company, bypassing the registry.
        skip_failures: Report and continue past unreadable or unattributable
            files rather than aborting the batch.

    Returns:
        ``(documents, skipped)`` — pages ready for chunking, and the files left
        out with the reason for each, so the caller can print both.
    """
    documents: list[Document] = []
    skipped: list[SkippedFile] = []

    for pdf in iter_pdfs(directory):
        try:
            company = resolve_company(pdf, registry, override)
            pages = load_company_pdf(pdf, company, document_type=document_type)
        except AdvisorError as exc:
            if not skip_failures:
                raise
            logger.error("Skipped %s: %s", pdf.name, exc.message)
            skipped.append(SkippedFile(pdf, exc.message))
            continue

        logger.info("Loaded %s: %d pages as %s (%s)", pdf.name, len(pages), company.ticker, company.sector)
        documents.extend(pages)

    return documents, skipped


def companies_in(documents: Iterable[Document]) -> dict[str, int]:
    """Count chunks or pages per company ticker, for ingest reporting."""
    counts: dict[str, int] = {}
    for document in documents:
        ticker = str(document.metadata.get("company") or "unattributed")
        counts[ticker] = counts.get(ticker, 0) + 1
    return dict(sorted(counts.items()))
