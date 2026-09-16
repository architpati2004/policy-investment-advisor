"""Who a company document is about.

A policy chunk is useful with only a publisher attached. A company chunk is not:
the whole point of the company index is to answer "does this affect *my*
holdings", which means every chunk must say which company and which sector it
belongs to, using the same spelling the portfolio uses. Files as downloaded do
not cooperate — ``godrej consumer.pdf``, ``GODREJCP_AR_2025.pdf`` and
``Godrej Consumer Products Annual Report.pdf`` are the same company.

So companies are declared once, in ``data/companies/registry.json``, and
filenames are resolved against that declaration:

.. code-block:: json

    [
      {
        "ticker": "GODREJCP",
        "name": "Godrej Consumer Products",
        "sector": "FMCG",
        "aliases": ["godrej consumer", "gcpl"]
      }
    ]

The ticker is the identity that reaches chunk metadata, because it is the one
spelling that does not drift. Sector is free text, but consistency matters: it
is what Phase 8 matches a holding's sector against, and "FMCG" will not match
"Fmcg " on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from backend.config import Settings, get_settings
from backend.exceptions import InvalidRegistryError
from backend.logging_config import get_logger

logger = get_logger(__name__)

REGISTRY_FILENAME = "registry.json"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Fold a name or filename to a comparable form: lowercase, alphanumeric only.

    ``"Godrej Consumer Products Ltd."`` and ``"godrej_consumer_products_ltd"``
    both become ``"godrejconsumerproductsltd"``, so spelling, punctuation and
    separator style stop mattering.
    """
    return _NON_ALNUM.sub("", text.lower())


@dataclass(frozen=True)
class Company:
    """One company the portfolio can hold."""

    ticker: str
    name: str
    sector: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name, value in (("ticker", self.ticker), ("name", self.name), ("sector", self.sector)):
            if not value or not str(value).strip():
                raise InvalidRegistryError(f"company entry is missing {name}")

    @property
    def keys(self) -> tuple[str, ...]:
        """Every spelling that identifies this company, longest first.

        Longest first so that ``Godrej Properties`` wins over ``Godrej`` when a
        filename could match both — a short alias must never shadow a specific
        one.
        """
        candidates = {self.ticker, self.name, *self.aliases}
        folded = {normalise(candidate) for candidate in candidates if candidate}
        return tuple(sorted((key for key in folded if key), key=len, reverse=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Company:
        if not isinstance(payload, dict):
            raise InvalidRegistryError(f"expected an object per company, got {type(payload).__name__}")
        aliases = payload.get("aliases") or []
        if isinstance(aliases, str) or not isinstance(aliases, Iterable):
            raise InvalidRegistryError(f"'aliases' must be a list of strings, got {aliases!r}")
        return cls(
            ticker=str(payload.get("ticker", "")).strip(),
            name=str(payload.get("name", "")).strip(),
            sector=str(payload.get("sector", "")).strip(),
            aliases=tuple(str(alias).strip() for alias in aliases if str(alias).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "aliases": list(self.aliases),
        }


class CompanyRegistry:
    """The declared universe of companies, and how to find one from a filename."""

    def __init__(self, companies: Iterable[Company] = ()) -> None:
        self._companies: list[Company] = list(companies)
        tickers = [company.ticker.upper() for company in self._companies]
        duplicates = {ticker for ticker in tickers if tickers.count(ticker) > 1}
        if duplicates:
            raise InvalidRegistryError(f"duplicate tickers: {', '.join(sorted(duplicates))}")

    def __len__(self) -> int:
        return len(self._companies)

    def __iter__(self) -> Iterator[Company]:
        return iter(self._companies)

    @property
    def companies(self) -> list[Company]:
        return list(self._companies)

    def sectors(self) -> list[str]:
        """Every distinct sector represented, sorted."""
        return sorted({company.sector for company in self._companies})

    def by_ticker(self, ticker: str) -> Company | None:
        folded = normalise(ticker)
        for company in self._companies:
            if normalise(company.ticker) == folded:
                return company
        return None

    def resolve(self, text: str) -> Company | None:
        """Find the company a filename or free-text label refers to.

        Matching is containment either way round on the folded forms, so
        ``GODREJCP_AR_2025.pdf`` matches by ticker and ``godrej consumer.pdf``
        by alias. The longest matching key wins, so a company whose name is a
        prefix of another's cannot steal its documents.
        """
        haystack = normalise(Path(text).stem if text.lower().endswith(".pdf") else text)
        if not haystack:
            return None

        best: tuple[int, Company] | None = None
        for company in self._companies:
            for key in company.keys:
                if key and (key in haystack or haystack in key):
                    if best is None or len(key) > best[0]:
                        best = (len(key), company)
                    break
        return best[1] if best else None

    # --- Persistence --------------------------------------------------------

    @classmethod
    def from_json(cls, payload: str) -> CompanyRegistry:
        try:
            data = json.loads(payload)
        except ValueError as exc:
            raise InvalidRegistryError(f"not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise InvalidRegistryError("the registry must be a JSON list of company objects")
        return cls(Company.from_dict(entry) for entry in data)

    @classmethod
    def load(cls, path: str | Path | None = None, settings: Settings | None = None) -> CompanyRegistry:
        """Read the registry from disk.

        A missing file is not an error: it yields an empty registry, so the CLI
        can still ingest documents whose company is named on the command line.
        """
        settings = settings or get_settings()
        path = Path(path) if path is not None else settings.company_data_dir / REGISTRY_FILENAME

        if not path.is_file():
            logger.info("No company registry at %s; falling back to command-line details", path)
            return cls()

        registry = cls.from_json(path.read_text(encoding="utf-8"))
        logger.info("Loaded %d companies from %s", len(registry), path)
        return registry

    def save(self, path: str | Path) -> None:
        """Write the registry back out, formatted for hand-editing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [company.to_dict() for company in self._companies]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- Matching against free text ----------------------------------------

    def mentioned_in(self, text: str) -> list[Company]:
        """Companies a piece of prose names, most specific first.

        Used for news, where the company is named in the body rather than in a
        filename, so the folded substring matching :meth:`resolve` uses would be
        far too loose — ``normalise`` strips spaces, and a ticker like
        ``RELIANCE`` would then match inside "reliance on imported crude".

        So prose is matched on word boundaries in the original text:

        * **name and aliases**, case-insensitively — "Godrej Consumer Products"
          and "godrej consumer" both count;
        * **the ticker**, only in upper case — news writes ``RELIANCE`` for the
          scrip and "reliance" for the noun, and the distinction is the only
          signal available.

        A company named after an ordinary word can still produce a false
        positive in its own name form; that is what ``aliases`` is for, and why
        attribution is reported per article rather than assumed.
        """
        if not text or not text.strip():
            return []

        matches: list[tuple[int, Company]] = []
        for company in self._companies:
            best = 0
            for phrase in (company.name, *company.aliases):
                if phrase and re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
                    best = max(best, len(phrase))
            if re.search(rf"\b{re.escape(company.ticker)}\b", text):
                best = max(best, len(company.ticker))
            if best:
                matches.append((best, company))

        matches.sort(key=lambda pair: pair[0], reverse=True)
        return [company for _, company in matches]
