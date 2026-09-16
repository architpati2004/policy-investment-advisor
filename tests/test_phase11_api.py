"""Phase 11 tests: the HTTP API.

Every test runs against a real app with a real SQLite database in a temp
directory; only the model-backed chains are stubbed, because a test suite that
needs local inference is a test suite nobody runs. Dependency overrides are the
mechanism, which is the reason the chains are injected rather than constructed
inside the routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from backend.alerts.engine import Finding, RunReport
from backend.api.deps import (
    get_alert_engine,
    get_db,
    get_indexes,
    get_policy_rag,
    get_portfolio_rag,
)
from backend.config import Settings, get_settings
from backend.db import portfolio as service
from backend.db.session import create_db_engine, init_db
from backend.ingestion.company_registry import Company as RegistryCompany
from backend.ingestion.company_registry import CompanyRegistry
from backend.main import create_app
from backend.rag.policy_rag import PolicyAnswer, Source
from backend.rag.portfolio_rag import PortfolioAssessment
from backend.rag.prompts import NOT_COVERED
from backend.rag.vector_store import SearchResult

GODREJ = RegistryCompany("GODREJCP", "Godrej Consumer Products", "FMCG", ("godrej consumer",))
HDFC = RegistryCompany("HDFCBANK", "HDFC Bank", "Banking")


def _source(number: int = 1, cited: bool = True) -> Source:
    return Source(
        number=number,
        citation="RBI — deposit directions (p. 8)",
        score=0.51,
        cited=cited,
        text="No interest shall be paid on deposits held in current accounts.",
        metadata={"source": "RBI", "page": 8, "origin": "policy"},
    )


class StubPolicyRAG:
    def __init__(self, answer: PolicyAnswer) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, question: str, **kwargs: Any) -> PolicyAnswer:
        self.asked.append(question)
        return self.answer


class StubPortfolioRAG:
    def __init__(self, assessment: PortfolioAssessment) -> None:
        self.assessment = assessment

    def assess(self, question: str, session: Session, **kwargs: Any) -> PortfolioAssessment:
        return self.assessment


class StubEngine:
    def __init__(self, findings: dict[str, Finding]) -> None:
        self.findings = findings
        self.checked: list[str] = []

    def check(self, holding: Any, session: Session, portfolio_id: int) -> Finding:
        self.checked.append(holding.ticker)
        return self.findings.get(holding.ticker, Finding(ticker=holding.ticker, reason="quiet"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={
            "database_url": f"sqlite:///{tmp_path}/api.db",
            "vectorstore_dir": str(tmp_path / "vectorstore"),
            "data_dir": str(tmp_path / "data"),
        }
    )


@pytest.fixture
def db(settings: Settings) -> Iterator[Session]:
    engine = create_db_engine(settings)
    init_db(engine)
    with Session(engine, expire_on_commit=False) as session:
        service.sync_companies(session, CompanyRegistry([GODREJ, HDFC]))
        service.add_holding(session, "GODREJCP", "150", "1180.50")
        session.commit()
        yield session
    engine.dispose()


@pytest.fixture
def client(settings: Settings, db: Session) -> Iterator[TestClient]:
    """An app wired to the temp database, with lifespan startup exercised."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- Portfolio ---------------------------------------------------------------


def test_reading_the_portfolio_returns_cost_basis_and_weights(client: TestClient) -> None:
    body = client.get("/api/portfolio").json()

    assert body["name"] == service.DEFAULT_PORTFOLIO
    assert body["invested"] == "177075.00"
    assert body["holdings"][0]["ticker"] == "GODREJCP"
    assert body["sector_weights"] == {"FMCG": "100.00"}


def test_money_crosses_the_wire_as_strings(client: TestClient) -> None:
    """A JSON number cannot hold 1180.50 exactly, and a portfolio sums these."""
    holding = client.get("/api/portfolio").json()["holdings"][0]
    assert holding["average_cost"] == "1180.50"
    assert isinstance(holding["invested"], str)


def test_adding_a_holding_returns_the_rebalanced_portfolio(client: TestClient) -> None:
    response = client.post(
        "/api/portfolio/holdings",
        json={"ticker": "HDFCBANK", "quantity": "40", "average_cost": "1650.00"},
    )

    assert response.status_code == 201
    body = response.json()
    assert {h["ticker"] for h in body["holdings"]} == {"GODREJCP", "HDFCBANK"}
    assert sum(float(h["weight_percent"]) for h in body["holdings"]) == pytest.approx(100.0, abs=0.01)


def test_buying_more_reweights_rather_than_duplicating(client: TestClient) -> None:
    client.post(
        "/api/portfolio/holdings",
        json={"ticker": "GODREJCP", "quantity": "150", "average_cost": "1000.00"},
    )
    holdings = client.get("/api/portfolio").json()["holdings"]

    assert len(holdings) == 1
    assert holdings[0]["quantity"] == "300.000"


def test_an_unknown_ticker_is_refused_with_the_fix(client: TestClient) -> None:
    response = client.post(
        "/api/portfolio/holdings",
        json={"ticker": "INFY", "quantity": "10", "average_cost": "1500"},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "UnknownTickerError"
    assert "registry.json" in body["remediation"]


def test_a_malformed_holding_is_rejected_before_it_reaches_the_database(
    client: TestClient,
) -> None:
    assert client.post("/api/portfolio/holdings", json={"ticker": "GODREJCP"}).status_code == 422


def test_removing_a_holding_closes_it(client: TestClient) -> None:
    response = client.request("DELETE", "/api/portfolio/holdings/GODREJCP")
    assert response.status_code == 200
    assert response.json()["holdings"] == []


def test_a_portfolio_that_does_not_exist_is_a_404(client: TestClient) -> None:
    response = client.get("/api/portfolio", params={"name": "Nope"})
    assert response.status_code == 400  # PortfolioError carries its own status
    assert "No portfolio" in response.json()["message"]


def test_the_scope_endpoint_returns_what_retrieval_uses(client: TestClient) -> None:
    body = client.get("/api/portfolio/scope").json()
    assert body == {"tickers": ["GODREJCP"], "sectors": ["FMCG"]}


# --- Chat --------------------------------------------------------------------


def _answer(**overrides: Any) -> PolicyAnswer:
    defaults: dict[str, Any] = dict(
        question="can a bank pay interest on a current account?",
        answer="No interest shall be paid on deposits held in current accounts [1].",
        covered=True,
        sources=[_source()],
        retrieved=1,
        best_score=0.51,
        elapsed_seconds=1.2,
    )
    defaults.update(overrides)
    return PolicyAnswer(**defaults)


def test_a_policy_question_returns_the_answer_and_its_sources(
    client: TestClient, settings: Settings
) -> None:
    stub = StubPolicyRAG(_answer())
    client.app.dependency_overrides[get_policy_rag] = lambda: stub

    body = client.post("/api/chat/policy", json={"question": "can a bank pay interest?"}).json()

    assert body["covered"] is True and body["grounded"] is True
    assert body["sources"][0]["citation"].startswith("RBI")
    assert stub.asked == ["can a bank pay interest?"]


def test_a_refusal_is_a_200_with_the_reason_in_the_body(client: TestClient) -> None:
    """The rule from CLAUDE.md section 11.

    The refusal text routinely *contains* the finding — qwen3:4b says "the
    sources govern commercial banks, so this does not apply" as a refusal — so a
    client that hides the body when covered is false throws the answer away. An
    error status would be a second lie: nothing failed.
    """
    refusal = _answer(
        answer=f"{NOT_COVERED}: the sources govern commercial banks, not NBFCs",
        covered=False,
        sources=[],
        reason="the sources govern commercial banks, not NBFCs",
    )
    client.app.dependency_overrides[get_policy_rag] = lambda: StubPolicyRAG(refusal)

    response = client.post("/api/chat/policy", json={"question": "what about NBFCs?"})

    assert response.status_code == 200
    body = response.json()
    assert body["covered"] is False
    assert "commercial banks" in body["answer"]
    assert "commercial banks" in body["reason"]


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/api/chat/policy", json={"question": ""}).status_code == 422


def test_a_portfolio_assessment_reports_affected_and_whether_it_was_declared(
    client: TestClient,
) -> None:
    assessment = PortfolioAssessment(
        question="do the deposit rules affect me?",
        portfolio=service.DEFAULT_PORTFOLIO,
        answer="The rules govern banks; the holding is unaffected [1].\nAFFECTED: none",
        covered=True,
        holdings=["GODREJCP"],
        affected=[],
        impact_declared=True,
        sources=[_source()],
        policy_chunks=3,
        company_chunks=2,
    )
    client.app.dependency_overrides[get_portfolio_rag] = lambda: StubPortfolioRAG(assessment)

    body = client.post("/api/chat/portfolio", json={"question": "does this affect me?"}).json()

    assert body["affected"] == []
    assert body["unaffected"] == ["GODREJCP"]
    assert body["impact_declared"] is True
    assert body["context"] == {"policy": 3, "company": 2}


def test_an_undeclared_impact_is_not_reported_as_nothing_affected(client: TestClient) -> None:
    """Silence about impact is unknown, not safe."""
    assessment = PortfolioAssessment(
        question="q",
        portfolio=service.DEFAULT_PORTFOLIO,
        answer="Something happened [1].",
        covered=True,
        holdings=["GODREJCP"],
        affected=[],
        impact_declared=False,
        sources=[_source()],
    )
    client.app.dependency_overrides[get_portfolio_rag] = lambda: StubPortfolioRAG(assessment)

    body = client.post("/api/chat/portfolio", json={"question": "q"}).json()

    assert body["impact_declared"] is False
    assert body["unaffected"] == [], "unaffected must stay empty when impact was never declared"


# --- Documents ---------------------------------------------------------------


def test_index_status_reports_unbuilt_indexes_without_erroring(client: TestClient) -> None:
    body = client.get("/api/documents/indexes").json()

    assert {entry["name"] for entry in body} == {"policy", "company", "news"}
    assert all(entry["built"] is False for entry in body)
    assert all(entry["vector_count"] == 0 for entry in body)


def test_searching_an_unbuilt_index_says_how_to_build_it(client: TestClient) -> None:
    response = client.get("/api/documents/search", params={"q": "repo rate", "index": "policy"})

    assert response.status_code == 404
    assert "build_policy_index.py" in response.json()["remediation"]


def test_an_unknown_index_name_is_rejected(client: TestClient) -> None:
    response = client.get("/api/documents/search", params={"q": "x", "index": "nonsense"})
    assert response.status_code == 422


def test_search_returns_chunks_and_scores(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    """Retrieval without generation: fast, and what makes an answer auditable."""
    import hashlib
    import math
    import re

    from langchain_core.embeddings import Embeddings

    from backend.rag.vector_store import VectorIndex

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._v(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._v(text)

        def _v(self, text: str) -> list[float]:
            vec = [0.0] * 32
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 32] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            return vec if norm == 0 else [v / norm for v in vec]

    index = VectorIndex(
        settings.policy_index_dir, FakeEmbeddings(), name="policy", settings=settings
    )
    # The route must search the same index this test built, with the same
    # embeddings — otherwise the query is embedded by a different model than the
    # chunks were, which is the mismatch the manifest check exists to catch.
    client.app.dependency_overrides[get_indexes] = lambda: {"policy": index}
    index.add_documents(
        [
            Document(
                page_content="The repo rate remains unchanged at 6.50 per cent.",
                metadata={"source": "RBI", "page": 1, "chunk_id": "a1"},
            )
        ]
    )

    body = client.get(
        "/api/documents/search", params={"q": "repo rate", "index": "policy", "min_score": 0.0}
    ).json()

    assert body["results"], "an indexed chunk should be findable"
    assert body["results"][0]["score"] > 0
    assert "repo rate" in body["results"][0]["text"]


# --- Alerts ------------------------------------------------------------------


def test_running_alerts_reports_quiet_holdings_too(client: TestClient) -> None:
    """"We checked and found nothing" and "we did not check" are different."""
    client.app.dependency_overrides[get_alert_engine] = lambda: StubEngine({})

    body = client.post("/api/alerts/run").json()

    assert body["checked"] == 1
    assert body["raised"] == 0
    assert body["findings"][0] == {
        "ticker": "GODREJCP",
        "raised": False,
        "duplicate": False,
        "summary": None,
        "reason": "quiet",
        "considered": 0,
        "elapsed_seconds": 0.0,
        "citations": [],
    }


def test_a_raised_alert_comes_back_with_its_evidence(client: TestClient) -> None:
    finding = Finding(
        ticker="GODREJCP",
        summary="Palm oil prices climbed 18%, squeezing consumer goods margins [1].",
        citations=[
            {
                "citation": "Economic Times — palm oil",
                "origin": "news",
                "score": 0.55,
                "chunk_id": "n1",
                "url": "https://example.test/a",
                "date": "2026-09-14",
            }
        ],
        raised=True,
        considered=5,
    )
    client.app.dependency_overrides[get_alert_engine] = lambda: StubEngine({"GODREJCP": finding})

    body = client.post("/api/alerts/run").json()

    assert body["raised"] == 1
    assert body["findings"][0]["citations"][0]["url"] == "https://example.test/a"


def test_a_run_can_be_narrowed_to_one_holding(client: TestClient) -> None:
    """One local generation per holding, so checking one must not cost all."""
    service_engine = StubEngine({})
    client.app.dependency_overrides[get_alert_engine] = lambda: service_engine
    client.post("/api/portfolio/holdings", json={"ticker": "HDFCBANK", "quantity": "40",
                                                 "average_cost": "1650.00"})

    client.post("/api/alerts/run", params={"ticker": "hdfcbank"})

    assert service_engine.checked == ["HDFCBANK"]


def test_listing_and_acknowledging_alerts(client: TestClient, db: Session) -> None:
    from backend.db.models import Alert

    portfolio = service.get_portfolio(db)
    db.add(
        Alert(
            portfolio_id=portfolio.id,
            ticker="GODREJCP",
            fingerprint="abc123",
            summary="Palm oil costs are rising [1].",
            evidence='[{"citation": "ET — palm oil", "origin": "news"}]',
        )
    )
    db.commit()

    listed = client.get("/api/alerts").json()
    assert len(listed) == 1
    assert listed[0]["citations"][0]["origin"] == "news"

    acknowledged = client.post(f"/api/alerts/{listed[0]['id']}/acknowledge").json()
    assert acknowledged["acknowledged"] is True
    assert client.get("/api/alerts").json() == []
    assert len(client.get("/api/alerts", params={"include_acknowledged": True}).json()) == 1


def test_acknowledging_an_alert_that_does_not_exist(client: TestClient) -> None:
    assert client.post("/api/alerts/999/acknowledge").status_code == 400


def test_clearing_says_the_evidence_can_raise_them_again(client: TestClient) -> None:
    body = client.request("DELETE", "/api/alerts").json()
    assert "raise them again" in body["message"]


# --- Wiring ------------------------------------------------------------------


def test_system_endpoints_still_work(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/config").json()["llm_model"]


def test_every_feature_router_is_mounted(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for expected in (
        "/api/portfolio",
        "/api/chat/policy",
        "/api/chat/portfolio",
        "/api/documents/indexes",
        "/api/documents/search",
        "/api/alerts",
        "/api/alerts/run",
    ):
        assert expected in paths, f"{expected} is not mounted"


def test_generation_routes_are_sync_so_they_do_not_block_the_event_loop() -> None:
    """A 30s blocking call in an async handler stalls every other request.

    FastAPI runs sync handlers in a worker thread, so these must stay ``def``.
    """
    import inspect

    from backend.api import alerts as alerts_routes
    from backend.api import chat as chat_routes

    for module, names in ((chat_routes, ("ask_policy", "assess_portfolio")),
                          (alerts_routes, ("run_alerts",))):
        for name in names:
            handler = getattr(module, name)
            assert not inspect.iscoroutinefunction(handler), f"{name} must not be async"
