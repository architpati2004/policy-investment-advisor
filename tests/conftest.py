"""Shared test fixtures.

The PDF builder lives in ``scripts/make_sample_pdf.py`` and is reused here, so
tests exercise real PDFs without the project taking on a PDF-generation
dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.make_sample_pdf import write_pdf


@pytest.fixture
def pdf_factory(tmp_path: Path):
    """Return a callable writing a PDF into the test's temp directory."""

    def _make(name: str, pages: list[list[str]]) -> Path:
        return write_pdf(tmp_path / name, pages)

    return _make
