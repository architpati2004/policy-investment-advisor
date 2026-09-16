"""The policy RAG chain: retrieve, ground, answer or refuse.

Composes Phase 4's index with the relevance policy and the grounding prompt into
the question-answering half of the architecture diagram. Portfolio awareness
arrives in Phase 8; this phase answers policy questions on their own.

Refusal is handled at two levels, because they catch different failures:

* **Before the model runs.** If nothing clears the absolute floor, the corpus is
  not about the question at all and there is nothing to ground an answer in. The
  chain says so without spending thirty seconds of local inference on it.
* **After the model runs.** If the retrieved chunks are topically close but
  govern something else, only a reader can tell. The model is asked to emit
  ``NOT_COVERED`` and the chain believes it.

Citations are verified rather than trusted: a ``[4]`` pointing at a source that
was never supplied is recorded as invalid instead of being resolved to whatever
chunk happens to be fourth.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.config import Settings, get_settings
from backend.logging_config import get_logger
from backend.rag import prompts
from backend.rag.ollama_client import build_chat_model, invoke_messages
from backend.rag.retrieval import PolicyRetriever, Retrieval
from backend.rag.vector_store import SearchResult, VectorIndex

logger = get_logger(__name__)

#: Citation markers the model is asked to produce: ``[1]``, ``[2]``.
CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Source:
    """A retrieved chunk offered to the model, and whether the answer used it."""

    number: int
    citation: str
    score: float
    cited: bool
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "citation": self.citation,
            "score": round(self.score, 4),
            "cited": self.cited,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicyAnswer:
    """A grounded answer, or a refusal, with everything needed to audit it."""

    question: str
    answer: str
    covered: bool
    sources: list[Source] = field(default_factory=list)
    reason: str | None = None
    invalid_citations: list[int] = field(default_factory=list)
    retrieved: int = 0
    best_score: float | None = None
    elapsed_seconds: float = 0.0
    generated: bool = True

    @property
    def cited_sources(self) -> list[Source]:
        """Only the sources the answer actually cites."""
        return [source for source in self.sources if source.cited]

    @property
    def grounded(self) -> bool:
        """True when an answer was given and it cites at least one source.

        A "covered" answer with no citation is the shape a hallucination takes
        here, so it is worth being able to ask about separately.
        """
        return self.covered and bool(self.cited_sources)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the CLI and the Phase 11 API."""
        return {
            "question": self.question,
            "answer": self.answer,
            "covered": self.covered,
            "grounded": self.grounded,
            "reason": self.reason,
            "sources": [source.to_dict() for source in self.sources],
            "invalid_citations": self.invalid_citations,
            "retrieved": self.retrieved,
            "best_score": round(self.best_score, 4) if self.best_score is not None else None,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def parse_citations(answer: str, available: int) -> tuple[set[int], list[int]]:
    """Split the ``[n]`` markers in an answer into valid and invalid ones.

    Args:
        answer: The model's reply.
        available: How many sources it was given.

    Returns:
        ``(valid, invalid)`` — numbers that name a real source, and numbers that
        do not. An invalid citation is a small hallucination and is reported
        rather than quietly resolved to some other chunk.
    """
    valid: set[int] = set()
    invalid: list[int] = []
    for match in CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= available:
            valid.add(number)
        elif number not in invalid:
            invalid.append(number)
    return valid, invalid


def _sources(results: list[SearchResult], cited: set[int]) -> list[Source]:
    return [
        Source(
            number=number,
            citation=result.citation(),
            score=result.score,
            cited=number in cited,
            text=result.text,
            metadata=dict(result.metadata),
        )
        for number, result in enumerate(results, 1)
    ]


class PolicyRAG:
    """Answers policy questions from the policy index, or declines to."""

    def __init__(
        self,
        retriever: PolicyRetriever | None = None,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
        *,
        index: VectorIndex | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.retriever = retriever or PolicyRetriever(index=index, settings=self._settings)
        # Built lazily so constructing the chain costs nothing and an unavailable
        # server is reported by the call that needs it, not by application start.
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = build_chat_model(self._settings)
        return self._chat_model

    def ask(self, question: str, *, k: int | None = None) -> PolicyAnswer:
        """Answer one policy question from the indexed corpus.

        Raises:
            IndexNotFoundError: the policy index has not been built.
            OllamaUnavailableError: the server went away during generation.
            LLMTimeoutError: generation exceeded ``LLM_TIMEOUT_SECONDS``.
        """
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        retrieval = self.retriever.retrieve(question, k=k)
        if not retrieval.has_context:
            return self._uncovered(question, retrieval, started)

        messages = prompts.build_messages(question, retrieval.results)
        logger.info(
            "Answering %r from %d chunks (best %.3f)",
            question,
            len(retrieval.results),
            retrieval.best_score or 0.0,
        )
        reply = invoke_messages(self.chat_model, messages, self._settings).strip()
        elapsed = time.perf_counter() - started

        if prompts.is_refusal(reply):
            detail = prompts.refusal_detail(reply)
            logger.info("Model declined to answer: %s", detail)
            return PolicyAnswer(
                question=question,
                # Prose only. `covered` carries the refusal, `reason` the detail,
                # so the sentinel itself has nothing left to say to a reader.
                answer=prompts.strip_sentinels(reply) or detail,
                covered=False,
                sources=_sources(retrieval.results, set()),
                reason=detail,
                retrieved=len(retrieval.results),
                best_score=retrieval.best_score,
                elapsed_seconds=elapsed,
            )

        cited, invalid = parse_citations(reply, len(retrieval.results))
        if invalid:
            logger.warning("Answer cited sources that were not supplied: %s", invalid)
        if not cited:
            logger.warning("Answer cites no source; treat it as unverified")

        return PolicyAnswer(
            question=question,
            answer=prompts.strip_sentinels(reply) or reply,
            covered=True,
            sources=_sources(retrieval.results, cited),
            invalid_citations=invalid,
            retrieved=len(retrieval.results),
            best_score=retrieval.best_score,
            elapsed_seconds=elapsed,
        )

    def _uncovered(self, question: str, retrieval: Retrieval, started: float) -> PolicyAnswer:
        """Refuse before generating: nothing retrieved is worth grounding in."""
        reason = retrieval.reason or "no relevant policy documents were retrieved"
        return PolicyAnswer(
            question=question,
            answer=reason,
            covered=False,
            reason=reason,
            retrieved=0,
            best_score=retrieval.best_score,
            elapsed_seconds=time.perf_counter() - started,
            generated=False,
        )


def build_policy_rag(settings: Settings | None = None) -> PolicyRAG:
    """Construct the chain from configuration. Contacts nothing."""
    return PolicyRAG(settings=settings or get_settings())
