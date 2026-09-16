"""Phase 6 tests: company registry, company ingestion and scoped retrieval.

Uses the deterministic in-process embedding model from Phase 4, so real FAISS
indexes are built on real temporary directories with no Ollama running.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config import Settings, get_settings
from backend.exceptions import InvalidRegistryError, UnknownCompanyError
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.company_docs import (
    companies_in,
    company_metadata,
    load_company_directory,
    load_company_pdf,
    resolve_company,
)
from backend.ingestion.company_registry import (
    Company,
    CompanyRegistry,
    normalise,
)
from backend.rag.retrieval import CompanyRetriever, RetrievalPolicy, scope_filter
from backend.rag.vector_store import VectorIndex, company_index

WORD = re.compile(r"[a-z0-9]+")


GODREJ = Company("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer", "gcpl"))
PROPERTIES = Company("GODREJPROP", "Godrej Properties", "Real Estate")
HDFC = Company("HDFCBANK", "HDFC Bank", "Banking")


@pytest.fixture
def registry() -> CompanyRegistry:
    return CompanyRegistry([GODREJ, PROPERTIES, HDFC])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={"vectorstore_dir": str(tmp_path / "vectorstore"), "data_dir": str(tmp_path / "data")}
    )


@pytest.fixture
def index(tmp_path: Path, settings: Settings) -> VectorIndex:
    return VectorIndex(
        tmp_path / "vectorstore" / "company_index",
        FakeEmbeddings(),
        name="company",
        settings=settings,
    )


def _chunk(text: str, company: Company, page: int = 1) -> Document:
    metadata = {
        "source": company.name,
        "document_type": "annual_report",
        "company": company.ticker,
        "sector": company.sector,
        "title": f"{company.ticker}_annual_report_2025",
        "page": page,
        "file_path": f"/data/companies/{company.ticker}.pdf",
        "chunk_id": f"{company.ticker}-{page}-{abs(hash(text)) % 10_000}",
    }
    return Document(page_content=text, metadata=metadata)


# --- Registry ----------------------------------------------------------------


def test_normalise_folds_spelling_punctuation_and_separators() -> None:
    assert normalise("Godrej Consumer Products Ltd.") == "godrejconsumerproductsltd"
    assert normalise("godrej_consumer_products_ltd") == "godrejconsumerproductsltd"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("godrej consumer.pdf", "GODREJCP"),
        ("GODREJCP_AR_2025.pdf", "GODREJCP"),
        ("gcpl.pdf", "GODREJCP"),
        ("Godrej Consumer Products Ltd Annual Report.pdf", "GODREJCP"),
        ("HDFCBANK-q3.pdf", "HDFCBANK"),
    ],
)
def test_registry_resolves_the_names_files_actually_arrive_with(
    registry: CompanyRegistry, filename: str, expected: str
) -> None:
    resolved = registry.resolve(filename)
    assert resolved is not None and resolved.ticker == expected


def test_a_longer_name_wins_over_a_shorter_one_it_contains(registry: CompanyRegistry) -> None:
    """'Godrej Properties' must not lose its filings to 'Godrej Consumer'."""
    resolved = registry.resolve("Godrej Properties Annual Report 2025.pdf")
    assert resolved is not None and resolved.ticker == "GODREJPROP"


def test_an_unknown_company_resolves_to_nothing(registry: CompanyRegistry) -> None:
    assert registry.resolve("infosys_annual_report.pdf") is None
    assert registry.resolve("") is None


def test_registry_round_trips_through_json(tmp_path: Path, registry: CompanyRegistry) -> None:
    path = tmp_path / "registry.json"
    registry.save(path)
    reloaded = CompanyRegistry.load(path)

    assert len(reloaded) == 3
    assert {c.ticker for c in reloaded} == {"GODREJCP", "GODREJPROP", "HDFCBANK"}
    assert reloaded.by_ticker("godrejcp") == GODREJ  # lookup is case-insensitive
    assert reloaded.sectors() == ["Banking", "FMCG", "Real Estate"]


def test_a_missing_registry_is_not_an_error(tmp_path: Path) -> None:
    """The CLI can still ingest with --company, so absence must not raise."""
    empty = CompanyRegistry.load(tmp_path / "nothing.json")
    assert len(empty) == 0
    assert empty.resolve("anything.pdf") is None


def test_malformed_registries_say_what_to_fix(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"

    path.write_text("{ not json")
    with pytest.raises(InvalidRegistryError):
        CompanyRegistry.load(path)

    path.write_text(json.dumps({"ticker": "X"}))  # object, not list
    with pytest.raises(InvalidRegistryError) as exc:
        CompanyRegistry.load(path)
    assert "registry.json" in str(exc.value)

    path.write_text(json.dumps([{"ticker": "X", "name": "X Ltd"}]))  # no sector
    with pytest.raises(InvalidRegistryError):
        CompanyRegistry.load(path)


def test_duplicate_tickers_are_rejected() -> None:
    with pytest.raises(InvalidRegistryError):
        CompanyRegistry([GODREJ, Company("GODREJCP", "Something Else", "IT")])


# --- Ingestion ---------------------------------------------------------------


def test_company_metadata_cites_the_name_and_keys_on_the_ticker() -> None:
    data = company_metadata(GODREJ, document_type="annual_report").to_dict()
    assert data["source"] == "Godrej Consumer Products"  # what a citation shows
    assert data["company"] == "GODREJCP"                  # what a holding matches
    assert data["sector"] == "FMCG"
    assert data["document_type"] == "annual_report"


def test_loading_a_filing_attaches_company_identity_to_every_page(pdf_factory) -> None:
    path = pdf_factory(
        "godrej consumer.pdf",
        [["Revenue from operations grew twelve per cent over the prior financial year."],
         ["Advertising spend increased as input cost inflation eased across categories."]],
    )
    pages = load_company_pdf(path, GODREJ)

    assert len(pages) == 2
    assert all(page.metadata["company"] == "GODREJCP" for page in pages)
    assert all(page.metadata["sector"] == "FMCG" for page in pages)
    assert [page.metadata["page"] for page in pages] == [1, 2]


def test_an_unattributable_file_is_refused_not_indexed_blindly(
    registry: CompanyRegistry, tmp_path: Path
) -> None:
    """A chunk with no company can never be matched to a holding, only stumbled on."""
    with pytest.raises(UnknownCompanyError) as exc:
        resolve_company(tmp_path / "some_unknown_firm.pdf", registry)
    assert "registry.json" in str(exc.value)


def test_an_override_attributes_a_file_the_registry_does_not_know(
    registry: CompanyRegistry, tmp_path: Path
) -> None:
    resolved = resolve_company(tmp_path / "mystery.pdf", registry, override=HDFC)
    assert resolved.ticker == "HDFCBANK"


def test_directory_load_reports_what_it_skipped(pdf_factory, registry, tmp_path: Path) -> None:
    pdf_factory("godrej consumer.pdf", [["Revenue from operations grew twelve per cent this year."]])
    pdf_factory("unknown_firm.pdf", [["Some other company's annual report text goes here today."]])

    pages, skipped = load_company_directory(tmp_path, registry)

    assert len(pages) == 1
    assert pages[0].metadata["company"] == "GODREJCP"
    assert len(skipped) == 1
    assert skipped[0].path.name == "unknown_firm.pdf"
    assert "which company" in skipped[0].reason


def test_directory_load_can_fail_fast(pdf_factory, registry, tmp_path: Path) -> None:
    pdf_factory("unknown_firm.pdf", [["Text belonging to nobody the registry has heard of."]])
    with pytest.raises(UnknownCompanyError):
        load_company_directory(tmp_path, registry, skip_failures=False)


def test_companies_in_counts_per_ticker() -> None:
    documents = [_chunk("a", GODREJ), _chunk("b", GODREJ, 2), _chunk("c", HDFC)]
    assert companies_in(documents) == {"GODREJCP": 2, "HDFCBANK": 1}


def test_chunks_keep_company_identity_through_chunking(pdf_factory) -> None:
    path = pdf_factory("gcpl.pdf", [["Revenue from operations grew twelve per cent. " * 40]])
    chunks = chunk_documents(load_company_pdf(path, GODREJ))

    assert len(chunks) > 1
    assert all(chunk.metadata["company"] == "GODREJCP" for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)


# --- Scope filtering ---------------------------------------------------------


def test_scope_filter_is_a_union_of_companies_and_sectors() -> None:
    """A sector-wide change matters to a holder whether or not their company is named."""
    matches = scope_filter(companies=["GODREJCP"], sectors=["Banking"])
    assert matches is not None
    assert matches({"company": "GODREJCP", "sector": "FMCG"})     # by company
    assert matches({"company": "HDFCBANK", "sector": "Banking"})  # by sector
    assert not matches({"company": "INFY", "sector": "IT"})


def test_scope_filter_ignores_capitalisation() -> None:
    matches = scope_filter(companies=["godrejcp"], sectors=["fmcg"])
    assert matches is not None
    assert matches({"company": "GODREJCP", "sector": "FMCG"})


def test_no_scope_means_no_filter() -> None:
    assert scope_filter(None, None) is None
    assert scope_filter([], []) is None
    assert scope_filter(["  "], None) is None


def test_scope_filter_does_not_match_missing_metadata() -> None:
    matches = scope_filter(companies=["GODREJCP"])
    assert matches is not None
    assert not matches({"source": "RBI"})
    assert not matches({"company": "", "sector": ""})


# --- Scoped retrieval --------------------------------------------------------


REVENUE = "Revenue from operations grew twelve per cent driven by volume growth in home care."
MARGIN = "Gross margin expanded ninety basis points as palm oil input costs moderated."
BANK = "Net interest margin was steady at four per cent with deposit growth of fifteen per cent."
REALTY = "Booking value rose across residential projects in the Mumbai metropolitan region."


#: Scope tests care about *which* chunks are reachable, not how relevant they
#: are. The configured floors are calibrated to embeddinggemma's score scale and
#: the deterministic test embedding does not share it, so a floor here would
#: reject perfectly good matches for reasons unrelated to what is being tested.
#: The floors have their own tests in Phase 5.
SCOPE_ONLY = RetrievalPolicy(top_k=5, min_score=0.0, relative_ratio=0.0)


def _retriever(index: VectorIndex, settings: Settings) -> CompanyRetriever:
    return CompanyRetriever(index=index, settings=settings, policy=SCOPE_ONLY)


@pytest.fixture
def populated(index: VectorIndex, settings: Settings) -> VectorIndex:
    index.add_documents(
        [
            _chunk(REVENUE, GODREJ, 1),
            _chunk(MARGIN, GODREJ, 2),
            _chunk(BANK, HDFC, 1),
            _chunk(REALTY, PROPERTIES, 1),
        ]
    )
    return index


def test_retrieval_scoped_to_a_company_returns_only_that_company(
    populated: VectorIndex, settings: Settings
) -> None:
    retriever = _retriever(populated, settings)
    retrieval = retriever.retrieve("growth", companies=["GODREJCP"])

    assert retrieval.has_context
    assert {r.metadata["company"] for r in retrieval.results} == {"GODREJCP"}


def test_retrieval_scoped_to_a_sector_spans_its_companies(
    index: VectorIndex, settings: Settings
) -> None:
    """Sector scope is how a policy change reaches holdings it never names."""
    marico = Company("MARICO", "Marico", "FMCG")
    index.add_documents([_chunk(REVENUE, GODREJ), _chunk(MARGIN, marico), _chunk(BANK, HDFC)])

    retriever = _retriever(index, settings)
    retrieval = retriever.retrieve("revenue growth and gross margin", sectors=["FMCG"], k=4)

    assert {r.metadata["company"] for r in retrieval.results} == {"GODREJCP", "MARICO"}


def test_unscoped_retrieval_sees_every_company(populated: VectorIndex, settings: Settings) -> None:
    retriever = _retriever(populated, settings)
    retrieval = retriever.retrieve("growth", k=4)
    assert len(retrieval.results) >= 1


def test_a_scope_with_nothing_in_it_says_so(populated: VectorIndex, settings: Settings) -> None:
    """'Nothing relevant' and 'nothing in scope' are different problems."""
    retriever = _retriever(populated, settings)
    retrieval = retriever.retrieve("growth", companies=["INFY"])

    assert not retrieval.has_context
    assert retrieval.reason and "INFY" in retrieval.reason


def test_a_filtered_search_sweeps_wide_enough_to_find_a_small_holding(
    index: VectorIndex, settings: Settings
) -> None:
    """FAISS filters after searching, so a narrow scope needs a wide sweep.

    One HDFC chunk hidden among two hundred FMCG ones is exactly the shape that
    a small fetch_k loses: every nearest neighbour belongs to somebody else.
    """
    bulk = [_chunk(f"{REVENUE} paragraph {n}", GODREJ, page=n) for n in range(200)]
    index.add_documents(bulk + [_chunk(REVENUE, HDFC, page=1)])

    retriever = _retriever(index, settings)
    retrieval = retriever.retrieve("revenue from operations growth", companies=["HDFCBANK"])

    assert retrieval.has_context, "the lone in-scope chunk was lost behind other companies"
    assert {r.metadata["company"] for r in retrieval.results} == {"HDFCBANK"}


def test_company_index_factory_points_at_the_configured_directory(settings: Settings) -> None:
    built = company_index(FakeEmbeddings(), settings)
    assert built.directory == settings.company_index_dir
    assert built.name == "company"


def test_citations_name_the_company_not_the_filename(populated: VectorIndex, settings: Settings) -> None:
    retriever = _retriever(populated, settings)
    citation = retriever.retrieve("growth", companies=["GODREJCP"]).results[0].citation()
    assert "Godrej Consumer Products" in citation
