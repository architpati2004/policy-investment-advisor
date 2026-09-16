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

import re
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
        elif chunk.origin == "news":
            # Labelled so the model can weigh a dated report differently from a
            # standing rule; the date is already appended below.
            label = "NEWS"
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


# --- Alerts (Phase 10) -------------------------------------------------------

#: Exact token the model emits when nothing in the sources is material.
NO_ALERT = "NO_ALERT"

#: Deliberately terse. An earlier draft spelled out what does not count as a
#: development — routine movement, another company, general commentary — and
#: qwen3:4b spent 2,260 tokens weighing those clauses before timing out at 300s.
#: Prompt wording drives token count here far more than context does (see the
#: Phase 8 table in CLAUDE.md section 11), so the judgement is stated once and
#: the disqualifying cases are left to the word "materially" and to the code,
#: which discards uncited and filing-only alerts regardless of what is said here.
ALERT_SYSTEM_PROMPT = f"""\
Report a development in the numbered sources that materially affects this holding.
Cite it as [1], [2]; an uncited report is discarded.
A HOLDING source is the company's own filing — evidence for why a development \
matters, never the development itself.
If there is none, reply exactly: {NO_ALERT}
One or two sentences. No preamble, no reasoning.
"""

ALERT_TEMPLATE = """\
Holding: {holding}

Sources:
{sources}

Does anything here materially affect this holding?"""


def build_alert_messages(holding: Any, chunks: list[Any]) -> list[tuple[str, str]]:
    """Assemble the chat messages for one holding's alert check."""
    description = (
        f"{holding.ticker} ({holding.name}), sector {holding.sector}, "
        f"{holding.weight}% of portfolio cost"
    )
    return [
        ("system", ALERT_SYSTEM_PROMPT),
        (
            "human",
            ALERT_TEMPLATE.format(holding=description, sources=format_context(chunks)),
        ),
    ]


def is_no_alert(reply: str) -> bool:
    """True when the model found nothing material.

    Checked per line, like :func:`is_refusal`: a model asked for a sentinel and a
    sentence will sometimes order them the other way round, and a parser that
    depends on the ordering eventually reads "nothing to report" as a warning.
    """
    return any(line.strip().upper().startswith(NO_ALERT) for line in reply.splitlines())


#: An alert summary needs at least this many words of prose once citation
#: markers are removed. qwen3:4b, asked tersely enough to finish inside the
#: deadline, answered one case with literally "[1], [2]" — correct citations,
#: correct decision to raise, and nothing a reader could act on.
#:
#: Deliberately low. The bar is "names something", not "writes at length": at
#: five this rejected "Risk weights rise", which is terse but perfectly
#: actionable, and a guard that polices style rather than emptiness would throw
#: away real alerts.
MIN_SUMMARY_WORDS = 3

_CITATION_MARKER = re.compile(r"\[\d{1,2}\]")
_WORD = re.compile(r"[A-Za-z]{2,}")


def has_substantive_prose(summary: str) -> bool:
    """True when a summary says something beyond pointing at sources.

    Strips citation markers and counts real words. The failure this exists for
    is not hypothetical: a model told to be terse will discover that citation
    markers alone technically satisfy "cite every claim", and an alert reading
    ``[1], [2]`` is worse than no alert, because it occupies a reader's
    attention while telling them nothing.
    """
    without_markers = _CITATION_MARKER.sub(" ", summary or "")
    return len(_WORD.findall(without_markers)) >= MIN_SUMMARY_WORDS


#: Line prefixes that are machinery, not prose. Each is parsed into a structured
#: field — ``covered``, ``affected``, ``impact_declared`` — so leaving it in the
#: text shows a reader the levers instead of the answer.
SENTINEL_PREFIXES = (NOT_COVERED, AFFECTED_PREFIX, NO_ALERT)


def strip_sentinels(reply: str) -> str:
    """The prose a reader should see, without the markers meant for the parser.

    Stripped *after* parsing, never before: the sentinels are how ``covered``,
    ``affected`` and ``impact_declared`` are determined, so removing them earlier
    would discard the meaning rather than relocate it.

    Line-scoped, matching how the parsers read them. Returns an empty string when
    a reply is nothing but markers, which lets callers substitute something a
    reader can use.
    """
    kept = [
        line
        for line in reply.splitlines()
        if not line.strip().upper().startswith(SENTINEL_PREFIXES)
    ]
    return "\n".join(kept).strip()


def strip_alert_markers(reply: str) -> str:
    """Alert summary without its sentinels, falling back to the raw reply."""
    return strip_sentinels(reply) or reply.strip()
