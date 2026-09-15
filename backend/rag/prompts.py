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
#: Line prefix the model ends a portfolio assessment with. Parsed in code for
#: the same reason as NOT_COVERED: reading impact out of prose cannot tell
#: "GODREJCP is exposed" from "GODREJCP is unaffected", and both name the
#: ticker. A declared line can say "none" and mean it.
AFFECTED_PREFIX = "AFFECTED:"

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


def _refusal_line(answer: str) -> str | None:
    """The line carrying the refusal sentinel, wherever the model put it."""
    for line in answer.splitlines():
        if line.strip().upper().startswith(NOT_COVERED):
            return line
    return None


def is_refusal(answer: str) -> bool:
    """True when the model declined to answer from the sources.

    Checked per line rather than only at the very start of the reply. Asked for
    both a refusal sentinel and a closing ``AFFECTED:`` line, qwen3:1.7b emitted
    them in the opposite order — ``AFFECTED: none`` first, the refusal second —
    and a reply that merely *starts* with something else is still a refusal. A
    parser that depends on the model's ordering will eventually read a refusal
    as an answer, which is the one direction this must never fail in.
    """
    return _refusal_line(answer) is not None


def refusal_detail(answer: str) -> str:
    """The model's one-line explanation of what the sources were missing."""
    line = _refusal_line(answer)
    if line is None:
        return "the retrieved sources do not cover this question"
    _, _, detail = line.partition(":")
    return detail.strip() or "the retrieved sources do not cover this question"


# --- Portfolio impact (Phase 8) ----------------------------------------------

PORTFOLIO_SYSTEM_PROMPT = f"""\
You assess how regulation and company disclosure bear on one specific \
portfolio, strictly from the numbered sources.
Cite every claim as [1], [2], and name affected holdings by ticker.
Sources bind only the institutions, instruments and companies they name. If \
nothing in the sources bears on a holding, say so and cite the sources that \
show it, rather than inferring a connection.
Reply {NOT_COVERED}: followed by what is missing only when the sources do not \
address the question at all.
Answer in one to four full sentences, never a bare verdict: no preamble, no \
reasoning, no restating the question.
End with one final line, exactly: {AFFECTED_PREFIX} followed by the tickers the \
sources show are affected, comma separated, or the word none.
"""

PORTFOLIO_TEMPLATE = """\
Portfolio:
{portfolio}

Sources:
{sources}

Question: {question}"""


def format_holdings(holdings: list[Any]) -> str:
    """Render the portfolio for the prompt.

    Ticker, name and sector, because all three are how a source can connect to a
    holding: a filing names the company, while a regulation usually names only
    an industry. Weights are included so "which of my holdings" can be answered
    with some sense of proportion, and are cost-basis shares, not valuations.
    """
    if not holdings:
        return "(no holdings)"
    return "\n".join(
        f"- {holding.ticker} ({holding.name}), sector {holding.sector}, "
        f"{holding.weight}% of cost"
        for holding in holdings
    )


def format_context(chunks: list[Any]) -> str:
    """Render merged context, labelled by which index each chunk came from.

    The label is what lets the model line a rule up against a holding: a source
    marked ``HOLDING GODREJCP`` is the company's own disclosure, while ``POLICY``
    is a regulator writing about an industry that may or may not include it.
    """
    rendered = []
    for number, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata
        if chunk.origin == "holding":
            label = f"HOLDING {metadata.get('company', 'unknown')}"
        else:
            label = "POLICY"
        header = f"[{number}] {label} — {display_title(metadata)}"
        page = metadata.get("page")
        if page is not None:
            header = f"{header}, page {page}"
        date = metadata.get("date")
        if date:
            header = f"{header}, dated {date}"
        rendered.append(f"{header}\n{' '.join(chunk.result.text.split())}")
    return "\n\n".join(rendered)


def build_portfolio_messages(
    question: str,
    holdings: list[Any],
    chunks: list[Any],
) -> list[tuple[str, str]]:
    """Assemble the chat messages for one portfolio impact question."""
    return [
        ("system", PORTFOLIO_SYSTEM_PROMPT),
        (
            "human",
            PORTFOLIO_TEMPLATE.format(
                portfolio=format_holdings(holdings),
                sources=format_context(chunks),
                question=question.strip(),
            ),
        ),
    ]
