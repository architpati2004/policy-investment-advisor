"""Phase 5 tests: relevance policy, grounding prompt and the policy RAG chain.

The default run uses stubs and touches neither Ollama nor FAISS, so it stays
fast. The live tests at the bottom are marked ``slow`` and excluded from the
default run (see ``pytest.ini``); run them with ``pytest -m slow``. They are the
ones that prove the behaviour that matters most: that the model refuses a
question the corpus cannot answer instead of inventing an answer from
near-identical rules written for somebody else.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from backend.config import Settings, get_settings
from backend.exceptions import LLMTimeoutError
from backend.rag import ollama_client, prompts
from backend.rag.policy_rag import PolicyRAG, parse_citations
from backend.rag.prompts import NOT_COVERED
from backend.rag.retrieval import (
    PolicyRetriever,
    Retrieval,
    RetrievalPolicy,
    select_relevant,
)
from backend.rag.vector_store import SearchResult, VectorIndex

# --- Helpers -----------------------------------------------------------------


def _result(score: float, text: str = "Some regulatory text.", **metadata: Any) -> SearchResult:
    meta = {
        "source": "RBI",
        "title": "RBI_commercial_banks_deposit_rate_directions_2025",
        "page": 6,
        "date": "2025-11-28",
        "chunk_id": f"chunk{int(score * 1000)}",
        **metadata,
    }
    return SearchResult(document=Document(page_content=text, metadata=meta), score=score)


class _Chunk:
    """One streamed chunk, as ``ChatOllama.stream`` yields them."""

    def __init__(self, content: str) -> None:
        self.content = content


class StubChatModel:
    """Returns a scripted reply and records what it was asked.

    Streams, because that is how the chain consumes a model: generation is read
    chunk by chunk so a wall-clock deadline can be enforced between chunks.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[tuple[str, str]]] = []

    def stream(self, messages: list[tuple[str, str]]) -> Any:
        self.calls.append(messages)
        # Split into two chunks so any accumulation bug shows up here.
        half = len(self.reply) // 2
        return iter([_Chunk(self.reply[:half]), _Chunk(self.reply[half:])])


class StubRetriever:
    """Returns a fixed retrieval without touching FAISS."""

    def __init__(self, retrieval: Retrieval) -> None:
        self.retrieval = retrieval

    def retrieve(self, question: str, *, k: int | None = None) -> Retrieval:
        return self.retrieval


def _retrieval(results: list[SearchResult], reason: str | None = None) -> Retrieval:
    return Retrieval(
        question="q",
        results=results,
        considered=len(results),
        best_score=results[0].score if results else None,
        floor=0.35,
        reason=reason,
    )


def _rag(reply: str, results: list[SearchResult], reason: str | None = None) -> tuple[PolicyRAG, StubChatModel]:
    model = StubChatModel(reply)
    rag = PolicyRAG(retriever=StubRetriever(_retrieval(results, reason)), chat_model=model)
    return rag, model


POLICY = RetrievalPolicy(top_k=5, min_score=0.35, relative_ratio=0.75)


# --- Relevance policy --------------------------------------------------------


def test_absolute_floor_rejects_a_question_this_corpus_is_not_about() -> None:
    """Measured: unrelated questions (GST, income tax) peak at 0.28."""
    kept, floor, reason = select_relevant([_result(0.28), _result(0.23)], POLICY)
    assert kept == []
    assert reason and "below the 0.35 floor" in reason
    assert floor == 0.35


def test_relative_floor_drops_the_tail_of_a_good_result_set() -> None:
    kept, floor, _ = select_relevant(
        [_result(0.58), _result(0.57), _result(0.56), _result(0.43)], POLICY
    )
    assert floor == pytest.approx(0.435)
    assert [round(r.score, 2) for r in kept] == [0.58, 0.57, 0.56]


def test_the_relative_floor_scales_with_the_corpus() -> None:
    """The point of the ratio: a fixed number tuned on 73 chunks would not.

    The same result *shape* at two different quality levels must keep the same
    members — the bar rises as the best match improves.
    """
    small = [_result(round(0.47 * f, 4)) for f in (1.0, 0.95, 0.80, 0.70)]
    large = [_result(round(0.80 * f, 4)) for f in (1.0, 0.95, 0.80, 0.70)]

    kept_small, floor_small, _ = select_relevant(small, POLICY)
    kept_large, floor_large, _ = select_relevant(large, POLICY)

    assert len(kept_small) == len(kept_large) == 3
    assert floor_large > floor_small


def test_a_weak_best_match_still_passes_if_it_clears_the_absolute_floor() -> None:
    """Measured: the weakest question the corpus genuinely answers scores 0.469.

    With a best match that weak the relative floor (0.469 x 0.75 = 0.352) falls
    to the absolute floor, so the two stop being separate filters and whatever
    clears 0.35 is kept. That is the intended interaction: the ratio tightens
    the net only when there is a strong match to measure against.
    """
    results = [_result(0.469), _result(0.401), _result(0.353)]
    kept, floor, reason = select_relevant(results, POLICY)

    assert reason is None
    assert floor == pytest.approx(max(POLICY.min_score, 0.469 * POLICY.relative_ratio))
    assert kept == sorted(results, key=lambda r: r.score, reverse=True)
    assert all(result.score >= floor for result in kept)


def test_empty_results_are_explained_not_crashed() -> None:
    kept, floor, reason = select_relevant([], POLICY)
    assert kept == [] and floor is None
    assert reason and "no chunks" in reason


def test_results_are_sorted_before_filtering() -> None:
    kept, _, _ = select_relevant([_result(0.40), _result(0.60), _result(0.50)], POLICY)
    assert [round(r.score, 2) for r in kept] == [0.60, 0.50]


def test_policy_reads_the_configured_floors() -> None:
    settings = get_settings()
    policy = RetrievalPolicy.from_settings(settings)
    assert policy.min_score == settings.min_relevance_score
    assert policy.relative_ratio == settings.relative_relevance_ratio
    assert policy.top_k == settings.retrieval_top_k


def test_retriever_scores_without_a_floor_so_the_reason_is_meaningful() -> None:
    """The 'why' message quotes the real best score, so retrieval must not pre-filter."""

    class RecordingIndex:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def search(self, question: str, **kwargs: Any) -> list[SearchResult]:
            self.kwargs = kwargs
            return [_result(0.28)]

    index = RecordingIndex()
    retrieval = PolicyRetriever(index=index).retrieve("what is the GST rate on cement?")
    assert index.kwargs["min_score"] == 0.0
    assert retrieval.best_score == pytest.approx(0.28)
    assert "0.28" in (retrieval.reason or "")


# --- Grounding prompt --------------------------------------------------------


def test_sources_are_numbered_from_one_to_match_citations() -> None:
    block = prompts.format_sources([_result(0.6, "First."), _result(0.5, "Second.")])
    assert block.startswith("[1]")
    assert "[2]" in block


def test_each_source_states_what_document_it_is() -> None:
    """Rule 2 can only work if the model can see what a source governs."""
    block = prompts.format_sources([_result(0.6)])
    assert "commercial banks" in block
    assert "page 6" in block
    assert "2025-11-28" in block


def test_display_title_makes_filenames_readable() -> None:
    assert prompts.display_title({"title": "RBI_commercial_banks_directions_2025"}) == (
        "RBI commercial banks directions 2025"
    )


def test_the_prompt_never_names_the_specific_trap_it_is_tested_on() -> None:
    """A prompt listing the test's answer would pass the test without helping."""
    text = prompts.POLICY_SYSTEM_PROMPT.lower()
    assert "nbfc" not in text
    assert "mutual fund" not in text
    assert "commercial bank" not in text
    # It must still demand the applicability judgement in general terms.
    assert "institution" in text and NOT_COVERED in prompts.POLICY_SYSTEM_PROMPT


def test_refusal_detection_reads_the_sentinel() -> None:
    assert prompts.is_refusal(f"{NOT_COVERED}: the sources cover banks only")
    assert prompts.is_refusal(f"  {NOT_COVERED.lower()}: whatever  ")
    assert not prompts.is_refusal("Banks cannot pay interest on current accounts [1].")


def test_refusal_detail_extracts_the_explanation() -> None:
    assert prompts.refusal_detail(f"{NOT_COVERED}: the sources govern banks") == (
        "the sources govern banks"
    )
    assert prompts.refusal_detail(NOT_COVERED)  # never empty


# --- Citation verification ---------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "available", "valid", "invalid"),
    [
        ("Yes [1] and [2].", 3, {1, 2}, []),
        ("Yes [4].", 3, set(), [4]),
        ("Yes [1] and [9].", 2, {1}, [9]),
        ("No citation at all.", 3, set(), []),
        ("Repeated [1] [1].", 1, {1}, []),
    ],
)
def test_parse_citations(answer: str, available: int, valid: set[int], invalid: list[int]) -> None:
    assert parse_citations(answer, available) == (valid, invalid)


# --- The chain ---------------------------------------------------------------


def test_refusal_before_the_model_runs_when_nothing_clears_the_floor() -> None:
    """A question this corpus is not about must not cost 30s of local inference."""
    rag, model = _rag("should never be called", [], reason="the closest chunk scored 0.28")

    answer = rag.ask("what is the GST rate on cement?")

    assert model.calls == []
    assert answer.covered is False
    assert answer.generated is False
    # `covered` carries the refusal; the text a reader sees is prose, not a
    # machine token they have to know how to read.
    assert NOT_COVERED not in answer.answer
    assert "0.28" in answer.answer
    assert "0.28" in (answer.reason or "")


def test_model_refusal_is_believed_and_cites_nothing() -> None:
    rag, _ = _rag(f"{NOT_COVERED}: the sources govern banks, not NBFCs", [_result(0.54)])

    answer = rag.ask("what deposit rates can an NBFC offer?")

    assert answer.covered is False
    assert answer.generated is True
    assert answer.cited_sources == []
    assert answer.reason == "the sources govern banks, not NBFCs"
    assert answer.retrieved == 1


def test_a_grounded_answer_reports_which_sources_it_used() -> None:
    rag, model = _rag("No interest is payable [1].", [_result(0.58), _result(0.50)])

    answer = rag.ask("can a bank pay interest on a current account?")

    assert answer.covered and answer.grounded
    assert [source.number for source in answer.cited_sources] == [1]
    assert answer.sources[1].cited is False
    assert answer.best_score == pytest.approx(0.58)
    assert answer.elapsed_seconds >= 0
    # The question and the sources both reached the model.
    system, human = model.calls[0]
    assert system[0] == "system" and NOT_COVERED in system[1]
    assert "current account" in human[1]


def test_an_answer_citing_a_source_that_was_never_supplied_is_flagged() -> None:
    rag, _ = _rag("It says so [3].", [_result(0.58)])

    answer = rag.ask("what do the directions say?")

    assert answer.invalid_citations == [3]
    assert answer.cited_sources == []
    assert answer.grounded is False  # covered, but nothing backs it


def test_an_uncited_answer_is_covered_but_not_grounded() -> None:
    rag, _ = _rag("Banks may not pay interest on current accounts.", [_result(0.58)])

    answer = rag.ask("can a bank pay interest?")

    assert answer.covered is True
    assert answer.grounded is False


def test_empty_questions_are_rejected() -> None:
    rag, _ = _rag("anything", [_result(0.58)])
    with pytest.raises(ValueError):
        rag.ask("   ")


def test_answer_serialises_for_the_api() -> None:
    import json

    rag, _ = _rag("Yes [1].", [_result(0.58)])
    payload = rag.ask("a question").to_dict()
    assert json.dumps(payload)
    assert payload["covered"] is True and payload["grounded"] is True
    assert payload["sources"][0]["cited"] is True


# --- Live tests (opt in with `pytest -m slow`) --------------------------------

BANK_DEPOSIT_TEXT = [
    "Chapter III – Domestic Rupee Deposits. A. Interest Rate on Domestic Current Account. "
    "8. No interest shall be paid on deposits held in current accounts by a commercial bank, "
    "provided that balances lying in the current account of a deceased individual depositor "
    "may earn interest at the rate applicable to savings deposits.",
    "9. Interest rates on domestic savings deposits offered by commercial banks shall be "
    "uniform across all branches and for all customers, and there shall be no discrimination "
    "between one depositor and another for deposits of the same amount and maturity.",
    "10. Penalties for premature withdrawal of a term deposit with a commercial bank shall be "
    "disclosed to the depositor at the time of accepting the deposit, and interest shall be "
    "paid at the rate applicable for the period the deposit actually remained with the bank.",
]


def _live_ready() -> bool:
    try:
        return ollama_client.check_status().ready
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(not _live_ready(), reason="Ollama or a configured model is unavailable")


@pytest.fixture
def live_rag(tmp_path: Path) -> PolicyRAG:
    """A real index of real commercial-bank deposit rules, and a real model.

    Self-contained rather than pointed at the project's own index, so the test
    proves the behaviour on any machine with Ollama running, not just one where
    the corpus happens to be built.
    """
    from backend.rag.embeddings import build_embeddings

    settings: Settings = get_settings().model_copy(
        update={"vectorstore_dir": str(tmp_path / "vectorstore")}
    )
    index = VectorIndex(
        tmp_path / "vectorstore" / "policy_index",
        build_embeddings(settings),
        name="policy",
        settings=settings,
    )
    index.add_documents(
        [
            Document(
                page_content=text,
                metadata={
                    "source": "RBI",
                    "title": "RBI_commercial_banks_deposit_rate_directions_2025",
                    "page": page,
                    "date": "2025-11-28",
                    "chunk_id": f"live{page}",
                },
            )
            for page, text in enumerate(BANK_DEPOSIT_TEXT, start=1)
        ]
    )
    return PolicyRAG(retriever=PolicyRetriever(index=index, settings=settings), settings=settings)


@pytest.mark.slow
@pytest.mark.integration
@live
def test_live_refuses_a_question_about_a_different_institution(live_rag: PolicyRAG) -> None:
    """The case no score can catch.

    These sources are about commercial-bank deposits; the question is about
    NBFCs. Retrieval scores this *above* several questions the corpus genuinely
    answers, so the refusal has to come from the model reading what the sources
    govern. Inventing NBFC rules out of bank rules is the failure this whole
    phase exists to prevent.
    """
    answer = live_rag.ask("what deposit rates can an NBFC offer to the public?")

    assert answer.covered is False, f"model answered instead of refusing: {answer.answer!r}"
    assert answer.generated is True, "this must be refused by the model, not by the score floor"
    assert answer.cited_sources == []
    assert answer.retrieved > 0


@pytest.mark.slow
@pytest.mark.integration
@live
def test_live_refuses_a_question_about_a_different_regulator(live_rag: PolicyRAG) -> None:
    answer = live_rag.ask("what are SEBI's mutual fund disclosure regulations?")
    assert answer.covered is False, f"model answered instead of refusing: {answer.answer!r}"


#: The prohibition, in whichever words the model reaches for: "no interest shall
#: be paid", "banks cannot pay interest", "interest may not be paid".
DENIES_INTEREST = re.compile(r"\b(no|not|cannot|shall not|may not)\b[^.]{0,60}\binterest\b", re.I)


@pytest.mark.slow
@pytest.mark.integration
@live
def test_live_answers_what_the_sources_do_cover(live_rag: PolicyRAG) -> None:
    """The happy path must still work — a system that refuses everything is useless.

    Asserted on grounding rather than wording: the answer has to cite the chunk
    that carries the rule, and has to convey the prohibition. Pinning the
    model's exact phrasing would test Qwen3's style, which can change without
    anything here being wrong.
    """
    answer = live_rag.ask("can a bank pay interest on a current account?")

    assert answer.covered is True, f"model refused an answerable question: {answer.answer!r}"
    assert answer.grounded is True, "an answer with no citation is not grounded"
    assert answer.invalid_citations == []
    cited = " ".join(source.text.lower() for source in answer.cited_sources)
    assert "current account" in cited, "cited a source that does not carry the rule"
    assert DENIES_INTEREST.search(answer.answer), f"answer does not state the rule: {answer.answer!r}"


@pytest.mark.integration
@live
def test_live_unrelated_question_is_refused_without_calling_the_model(live_rag: PolicyRAG) -> None:
    """Cheap half of the guard: no model call, so this stays in the default run."""
    answer = live_rag.ask("what is the GST rate on cement?")

    assert answer.covered is False
    assert answer.generated is False, "an off-topic question must not reach the model"
    assert math.isfinite(answer.best_score or 0.0)
    assert answer.elapsed_seconds < 10


# --- Generation deadline -----------------------------------------------------


class SlowStreamModel:
    """Streams chunks forever, as a wedged local generation does."""

    def __init__(self, chunks: int = 1000) -> None:
        self.chunks = chunks
        self.closed = False

    def stream(self, messages: list[tuple[str, str]]) -> Any:
        model = self

        class _Stream:
            def __iter__(self) -> Any:
                for index in range(model.chunks):
                    yield _Chunk(f"token{index} ")

            def close(self) -> None:
                model.closed = True

        return _Stream()


class FakeClock:
    """Monotonic clock that jumps a fixed amount on every reading."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def test_generation_exceeding_the_deadline_raises_timeout() -> None:
    """The bug this guards: httpx's timeout bounds the gap between tokens, not
    the call, so a generation that keeps streaming never trips it. A 227s answer
    ran past a 180s LLM_TIMEOUT_SECONDS before this deadline existed."""
    model = SlowStreamModel()
    settings = get_settings()

    with pytest.raises(LLMTimeoutError) as exc:
        ollama_client.invoke_messages(
            model, [("human", "hello")], settings, timeout_seconds=30, clock=FakeClock(step=20.0)
        )

    assert "30s" in str(exc.value)
    assert model.closed, "the stream must be closed so the generation stops costing CPU"


def test_generation_inside_the_deadline_returns_the_whole_text() -> None:
    model = SlowStreamModel(chunks=3)
    text = ollama_client.invoke_messages(
        model, [("human", "hello")], get_settings(), timeout_seconds=30, clock=FakeClock(step=0.1)
    )
    assert text == "token0 token1 token2 "
    assert model.closed


def test_a_stalled_stream_is_reported_as_a_timeout() -> None:
    """No first token at all: httpx's own read timeout fires, and is translated."""

    class StalledModel:
        def stream(self, messages: list[tuple[str, str]]) -> Any:
            raise RuntimeError("timed out waiting for response")

    with pytest.raises(LLMTimeoutError):
        ollama_client.invoke_messages(StalledModel(), [("human", "hi")], get_settings())


# --- Sentinels stay out of the text a reader sees ----------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("NOT_COVERED: the sources govern banks", ""),
        ("AFFECTED: none", ""),
        ("NO_ALERT", ""),
        ("Banks may not pay interest [1].\nAFFECTED: none", "Banks may not pay interest [1]."),
        ("AFFECTED: GODREJCP\nInput costs rose [2].", "Input costs rose [2]."),
        ("Plain prose with no markers.", "Plain prose with no markers."),
    ],
)
def test_strip_sentinels(reply: str, expected: str) -> None:
    assert prompts.strip_sentinels(reply) == expected


def test_a_model_refusal_reaches_the_caller_as_prose() -> None:
    """The sentinel is how `covered` is decided; it is not something to display."""
    rag, _ = _rag(f"{NOT_COVERED}: the sources govern banks, not NBFCs", [_result(0.54)])

    answer = rag.ask("what deposit rates can an NBFC offer?")

    assert answer.covered is False
    assert NOT_COVERED not in answer.answer
    assert answer.answer == "the sources govern banks, not NBFCs"
    assert answer.reason == "the sources govern banks, not NBFCs"


def test_an_answer_keeps_its_prose_when_the_model_adds_a_sentinel_line() -> None:
    rag, _ = _rag(f"No interest is payable [1].\n{NOT_COVERED}: partially", [_result(0.58)])

    answer = rag.ask("can a bank pay interest?")

    assert NOT_COVERED not in answer.answer
    assert "No interest is payable [1]." in answer.answer
