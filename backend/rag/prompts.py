"""The grounding prompt and the source block that feeds it.

This is where the system's central promise is enforced: an answer is either
supported by a retrieved chunk and cites it, or it is a refusal. Three details
carry that weight.

**The refusal sentinel.** The model replies ``NOT_COVERED: ...`` when the
sources do not answer the question. A fixed token is checked in code, which is
sturdier than pattern-matching prose like "I could not find" across phrasings.

**The applicability rule.** Scores cannot catch the dangerous case — a question
about one kind of institution retrieving near-identical rules written for
another (measured at 0.539, above three questions the corpus genuinely answers).
Rule 2 asks the model for the judgement the embedding cannot make. It is
deliberately phrased in general terms, naming no institution: a prompt that
listed the specific trap would teach the test its answer without making the
system any better at the traps nobody wrote down.

**Terseness, which turned out to be the whole latency story.** Qwen3 reasons
before answering, and how much it reasons tracks how deliberative the prompt
sounds, not how much context it gets. Measured on an M1, same question, same
chunks:

============================  ==========  ===========
prompt                        answerable  must refuse
============================  ==========  ===========
five numbered rules             2237 tok     1565 tok
these four lines                 259 tok      411 tok
============================  ==========  ===========

That is 265 s against 45 s, from wording alone — an earlier draft of this file,
saying the same things at five times the length, was the slowest configuration
measured in the whole project. Halving the *context* changed nothing (1896 tok
at five chunks, 1947 at two), so brevity here is a latency control, not a style
preference. Note also that the applicability line is not what costs tokens: it
*lowered* generation on answerable questions (512 -> 259) by giving the model a
rule to apply instead of a judgement to agonise over.

Disabling Qwen3's thinking outright was measured too, and is worse than either:
it does not stop the model reasoning, only stops Ollama separating the reasoning
out, so 1300-1900 characters of chain-of-thought land in the answer itself.
"""

from __future__ import annotations

from typing import Any

from backend.rag.vector_store import SearchResult

#: Exact token the model emits when the sources cannot answer the question.
NOT_COVERED = "NOT_COVERED"

POLICY_SYSTEM_PROMPT = f"""\
You answer strictly from the numbered sources. Cite as [1], [2].
Reply with the answer only: no preamble, no reasoning, no restating the question. \
Three sentences at most.
If the sources do not answer the question, begin your reply with {NOT_COVERED}: \
and then name in one sentence what is missing.
Sources bind only the institutions and instruments they name; a question about a \
different one is not covered.
"""

USER_TEMPLATE = """\
Sources:
{sources}

Question: {question}"""


def display_title(metadata: dict[str, Any]) -> str:
    """Readable document name for the source block.

    Titles come from filenames, so ``RBI_commercial_banks_deposit_rate_directions_2025``
    becomes ``RBI commercial banks deposit rate directions 2025``. The words
    matter: rule 2 asks the model to compare what a source governs against what
    the question asks, and it can only do that if the source says what it is.
    """
    title = str(metadata.get("title") or metadata.get("source") or "untitled")
    return title.replace("_", " ").strip()


def format_source(result: SearchResult, number: int) -> str:
    """Render one retrieved chunk as a numbered, attributed source."""
    metadata = result.metadata
    header = f"[{number}] {display_title(metadata)}"
    page = metadata.get("page")
    if page is not None:
        header = f"{header}, page {page}"
    date = metadata.get("date")
    if date:
        header = f"{header}, dated {date}"
    text = " ".join(result.text.split())
    return f"{header}\n{text}"


def format_sources(results: list[SearchResult]) -> str:
    """Render the whole source block, numbered from 1 to match citations."""
    return "\n\n".join(format_source(result, number) for number, result in enumerate(results, 1))


def build_messages(question: str, results: list[SearchResult]) -> list[tuple[str, str]]:
    """Assemble the chat messages for one grounded question."""
    return [
        ("system", POLICY_SYSTEM_PROMPT),
        ("human", USER_TEMPLATE.format(sources=format_sources(results), question=question.strip())),
    ]


def is_refusal(answer: str) -> bool:
    """True when the model declined to answer from the sources."""
    return answer.strip().upper().startswith(NOT_COVERED)


def refusal_detail(answer: str) -> str:
    """The model's one-line explanation of what the sources were missing."""
    text = answer.strip()
    _, _, detail = text.partition(":")
    return detail.strip() or "the retrieved sources do not cover this question"
