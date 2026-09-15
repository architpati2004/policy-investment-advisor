"""Phase 7 tests: portfolio schema and operations.

Every test runs against a real SQLite database in a temp directory, so foreign
keys, constraints and the integer money arithmetic are exercised for real rather
than mocked.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.db import portfolio as service
from backend.db.models import (
    Company,
    Holding,
    Portfolio,
    from_milli,
    from_paise,
    to_milli,
    to_paise,
)
from backend.db.session import create_db_engine, init_db
from backend.exceptions import PortfolioError, UnknownTickerError
from backend.ingestion.company_registry import Company as RegistryCompany
from backend.ingestion.company_registry import CompanyRegistry

GODREJ = RegistryCompany("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer",))
HDFC = RegistryCompany("HDFCBANK", "HDFC Bank", "Banking")
RELIANCE = RegistryCompany("RELIANCE", "Reliance Industries", "Energy")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"database_url": f"sqlite:///{tmp_path}/test.db"})


@pytest.fixture
def session(settings: Settings):
    """A session on a fresh database, rolled back and disposed afterwards."""
    engine = create_db_engine(settings)
    init_db(engine)
    with Session(engine, expire_on_commit=False) as open_session:
        yield open_session
    engine.dispose()


@pytest.fixture
def registry() -> CompanyRegistry:
    return CompanyRegistry([GODREJ])


@pytest.fixture
def seeded(session: Session, registry: CompanyRegistry) -> Session:
    service.sync_companies(session, registry)
    return session


# --- Money and quantity ------------------------------------------------------


@pytest.mark.parametrize(
    ("rupees", "paise"),
    [("1180.50", 118050), (1180.5, 118050), (0.1, 10), ("0.005", 1), (2499.99, 249999)],
)
def test_rupees_convert_to_whole_paise(rupees: object, paise: int) -> None:
    assert to_paise(rupees) == paise


def test_money_survives_a_round_trip_that_floats_would_spoil() -> None:
    """0.1 + 0.2 != 0.3 in binary floating point; in paise it is exact."""
    assert from_paise(to_paise("0.1") + to_paise("0.2")) == Decimal("0.30")


def test_fractional_share_counts_are_exact() -> None:
    assert from_milli(to_milli("0.125")) == Decimal("0.125")
    assert to_milli(150) == 150_000


def test_holding_exposes_decimals_not_integers(seeded: Session) -> None:
    holding = service.add_holding(seeded, "GODREJCP", "150", "1180.50")
    assert holding.quantity == Decimal("150.000")
    assert holding.average_cost == Decimal("1180.50")
    assert holding.invested == Decimal("177075.00")  # 150 x 1180.50, exactly


# --- Schema ------------------------------------------------------------------


def test_the_schema_is_three_tables(session: Session) -> None:
    tables = set(inspect(session.get_bind()).get_table_names())
    assert {"companies", "portfolios", "holdings"} <= tables


def test_a_company_cannot_be_held_twice_in_one_portfolio(seeded: Session) -> None:
    """Buying more must re-weight the average, not create a second row."""
    portfolio = service.get_or_create_portfolio(seeded)
    seeded.add(Holding(portfolio_id=portfolio.id, ticker="GODREJCP", quantity_milli=1, average_cost_paise=1))
    seeded.add(Holding(portfolio_id=portfolio.id, ticker="GODREJCP", quantity_milli=2, average_cost_paise=2))
    with pytest.raises(IntegrityError):
        seeded.flush()
    seeded.rollback()


def test_foreign_keys_are_actually_enforced(seeded: Session) -> None:
    """SQLite ignores foreign keys unless asked; an orphan holding must not exist."""
    portfolio = service.get_or_create_portfolio(seeded)
    seeded.add(
        Holding(portfolio_id=portfolio.id, ticker="NOTREAL", quantity_milli=1000, average_cost_paise=100)
    )
    with pytest.raises(IntegrityError):
        seeded.flush()
    seeded.rollback()


def test_deleting_a_portfolio_takes_its_holdings_with_it(seeded: Session) -> None:
    service.add_holding(seeded, "GODREJCP", "10", "1000")
    portfolio = service.get_portfolio(seeded)

    seeded.delete(portfolio)
    seeded.flush()

    assert seeded.scalars(select(Holding)).all() == []
    assert seeded.get(Company, "GODREJCP") is not None  # the company itself survives


# --- Companies ---------------------------------------------------------------


def test_sync_imports_the_registry(session: Session, registry: CompanyRegistry) -> None:
    added, updated = service.sync_companies(session, registry)
    assert (added, updated) == (1, 0)
    assert service.known_tickers(session) == ["GODREJCP"]


def test_sync_is_idempotent_and_picks_up_edits(session: Session) -> None:
    service.sync_companies(session, CompanyRegistry([GODREJ]))
    added, updated = service.sync_companies(session, CompanyRegistry([GODREJ]))
    assert (added, updated) == (0, 0)

    renamed = RegistryCompany("GODREJCP", "Godrej Consumer Products Ltd", "Consumer Staples")
    added, updated = service.sync_companies(session, CompanyRegistry([renamed]))
    assert (added, updated) == (0, 1)
    assert session.get(Company, "GODREJCP").sector == "Consumer Staples"


# --- Holdings ----------------------------------------------------------------


def test_adding_a_holding_records_quantity_cost_and_sector(seeded: Session) -> None:
    holding = service.add_holding(seeded, "GODREJCP", "150", "1180.50", bought_on=date(2026, 4, 1))

    assert holding.ticker == "GODREJCP"
    assert holding.sector == "FMCG"
    assert holding.first_bought_on == date(2026, 4, 1)


def test_tickers_are_normalised_to_upper_case(seeded: Session) -> None:
    holding = service.add_holding(seeded, " godrejcp ", "10", "1000")
    assert holding.ticker == "GODREJCP"


def test_buying_more_reweights_the_average_cost(seeded: Session) -> None:
    """100 at 1000 then 100 at 1200 averages to 1100, not 1000 or 1200."""
    service.add_holding(seeded, "GODREJCP", "100", "1000.00")
    holding = service.add_holding(seeded, "GODREJCP", "100", "1200.00")

    assert holding.quantity == Decimal("200.000")
    assert holding.average_cost == Decimal("1100.00")
    assert len(service.list_holdings(seeded)) == 1


def test_repeated_purchases_do_not_drift(seeded: Session) -> None:
    """Integer arithmetic means a hundred small buys stay exact."""
    for _ in range(100):
        service.add_holding(seeded, "GODREJCP", "1", "333.33")

    holding = service.list_holdings(seeded)[0]
    assert holding.quantity == Decimal("100.000")
    assert holding.average_cost == Decimal("333.33")
    assert holding.invested == Decimal("33333.00")


def test_an_earlier_purchase_date_wins(seeded: Session) -> None:
    service.add_holding(seeded, "GODREJCP", "10", "1000", bought_on=date(2026, 6, 1))
    holding = service.add_holding(seeded, "GODREJCP", "10", "1000", bought_on=date(2026, 1, 15))
    assert holding.first_bought_on == date(2026, 1, 15)


def test_holding_an_unknown_company_is_refused(seeded: Session) -> None:
    """No sector means it could never be matched to a policy change."""
    with pytest.raises(UnknownTickerError) as exc:
        service.add_holding(seeded, "INFY", "10", "1500")
    assert "registry.json" in str(exc.value)
    assert "GODREJCP" in str(exc.value)  # says what it does know


@pytest.mark.parametrize(("quantity", "cost"), [("0", "1000"), ("-5", "1000"), ("10", "0")])
def test_nonsense_positions_are_refused(seeded: Session, quantity: str, cost: str) -> None:
    with pytest.raises(PortfolioError):
        service.add_holding(seeded, "GODREJCP", quantity, cost)


def test_removing_a_position_closes_it(seeded: Session) -> None:
    service.add_holding(seeded, "GODREJCP", "150", "1180.50")
    service.remove_holding(seeded, "GODREJCP")
    assert service.list_holdings(seeded) == []


def test_removing_something_not_held_says_so(seeded: Session) -> None:
    service.add_holding(seeded, "GODREJCP", "10", "1000")
    with pytest.raises(PortfolioError) as exc:
        service.remove_holding(seeded, "HDFCBANK")
    assert "does not hold" in str(exc.value)


def test_asking_for_a_portfolio_that_does_not_exist_says_so(seeded: Session) -> None:
    with pytest.raises(PortfolioError):
        service.get_portfolio(seeded, "No Such Portfolio")


# --- Derived views -----------------------------------------------------------


def test_scope_is_the_shape_retrieval_takes(seeded: Session) -> None:
    """The join between what is owned and which documents are about it."""
    service.add_holding(seeded, "GODREJCP", "150", "1180.50")
    scope = service.portfolio_scope(seeded)

    assert scope.tickers == ("GODREJCP",)
    assert scope.sectors == ("FMCG",)
    assert not scope.is_empty


def test_scope_feeds_scope_filter_directly(seeded: Session) -> None:
    """Phase 8's hand-off, asserted now so the two halves cannot drift apart."""
    from backend.rag.retrieval import scope_filter

    service.add_holding(seeded, "GODREJCP", "150", "1180.50")
    scope = service.portfolio_scope(seeded)
    matches = scope_filter(companies=scope.tickers, sectors=scope.sectors)

    assert matches is not None
    assert matches({"company": "GODREJCP", "sector": "FMCG"})
    assert matches({"company": "MARICO", "sector": "FMCG"})  # sector reaches unheld peers
    assert not matches({"company": "HDFCBANK", "sector": "Banking"})


def test_an_empty_portfolio_has_an_empty_scope(seeded: Session) -> None:
    service.get_or_create_portfolio(seeded)
    assert service.portfolio_scope(seeded).is_empty


def test_weights_are_by_cost_and_sum_to_a_hundred(session: Session) -> None:
    service.sync_companies(session, CompanyRegistry([GODREJ, HDFC]))
    service.add_holding(session, "GODREJCP", "100", "1000.00")  # 100,000
    service.add_holding(session, "HDFCBANK", "100", "3000.00")  # 300,000

    summary = service.summarise(session)

    assert summary.invested == Decimal("400000.00")
    assert [h.ticker for h in summary.holdings] == ["HDFCBANK", "GODREJCP"]  # largest first
    assert [h.weight for h in summary.holdings] == [Decimal("75.00"), Decimal("25.00")]
    assert sum(h.weight for h in summary.holdings) == Decimal("100.00")


def test_sector_weights_combine_holdings_in_the_same_sector(session: Session) -> None:
    marico = RegistryCompany("MARICO", "Marico", "FMCG")
    service.sync_companies(session, CompanyRegistry([GODREJ, marico, HDFC]))
    service.add_holding(session, "GODREJCP", "100", "1000.00")
    service.add_holding(session, "MARICO", "100", "1000.00")
    service.add_holding(session, "HDFCBANK", "100", "2000.00")

    weights = service.summarise(session).sector_weights
    assert weights == {"Banking": Decimal("50.00"), "FMCG": Decimal("50.00")}


def test_summary_serialises_for_the_api(seeded: Session) -> None:
    import json

    service.add_holding(seeded, "GODREJCP", "150", "1180.50")
    payload = service.summarise(seeded).to_dict()

    assert json.dumps(payload)  # Decimals must not leak into the response
    assert payload["holdings"][0]["ticker"] == "GODREJCP"
    assert payload["sector_weights"] == {"FMCG": "100.00"}


# --- Extensibility -----------------------------------------------------------


def test_adding_hdfcbank_and_reliance_needs_no_schema_change(
    session: Session, registry: CompanyRegistry
) -> None:
    """The requirement stated for this phase, asserted rather than assumed.

    Start with the demo portfolio's single GODREJCP holding, then take the
    registry that a user would edit, add two companies from different sectors,
    and hold them. Nothing but rows changes: the table definitions before and
    after must be identical.
    """
    inspector = inspect(session.get_bind())

    def schema() -> dict[str, list[str]]:
        return {
            table: sorted(column["name"] for column in inspector.get_columns(table))
            for table in sorted(inspector.get_table_names())
        }

    service.seed_demo(session, registry)
    before = schema()
    assert service.portfolio_scope(session).tickers == ("GODREJCP",)

    # Exactly what a user does: declare the companies, sync, buy.
    service.sync_companies(session, CompanyRegistry([GODREJ, HDFC, RELIANCE]))
    service.add_holding(session, "HDFCBANK", "40", "1650.00")
    service.add_holding(session, "RELIANCE", "25", "2980.75")

    assert schema() == before, "holding a new company must not change the schema"

    scope = service.portfolio_scope(session)
    assert scope.tickers == ("GODREJCP", "HDFCBANK", "RELIANCE")
    assert scope.sectors == ("Banking", "Energy", "FMCG")
    assert len(service.summarise(session).holdings) == 3


def test_seed_demo_creates_the_illustrative_position(
    session: Session, registry: CompanyRegistry
) -> None:
    summary = service.seed_demo(session, registry)

    assert summary.name == service.DEFAULT_PORTFOLIO
    assert [h.ticker for h in summary.holdings] == ["GODREJCP"]
    assert summary.invested == Decimal("177075.00")


def test_seed_demo_skips_companies_the_registry_does_not_declare(session: Session) -> None:
    """So the demo list can name future holdings without breaking today."""
    summary = service.seed_demo(
        session,
        CompanyRegistry([GODREJ]),
        holdings=[("GODREJCP", "150", "1180.50"), ("HDFCBANK", "40", "1650.00")],
    )
    assert [h.ticker for h in summary.holdings] == ["GODREJCP"]


def test_a_second_portfolio_is_independent(session: Session, registry: CompanyRegistry) -> None:
    service.sync_companies(session, registry)
    service.add_holding(session, "GODREJCP", "100", "1000", portfolio_name="Demo Portfolio")
    service.add_holding(session, "GODREJCP", "50", "1200", portfolio_name="Second Portfolio")

    assert service.summarise(session, "Demo Portfolio").invested == Decimal("100000.00")
    assert service.summarise(session, "Second Portfolio").invested == Decimal("60000.00")
    assert session.scalars(select(Portfolio)).all().__len__() == 2
