import { useEffect, useState } from "react";
import { ErrorBanner, Empty } from "../components/Banner";
import { getIndexes, search } from "../services/documents";
import { ApiError } from "../services/client";
import type { IndexStatus, SearchResults } from "../types";

const BUILD_COMMAND: Record<string, string> = {
  policy: "python scripts/build_policy_index.py build",
  company: "python scripts/build_company_index.py build",
  news: "python scripts/build_news_index.py build",
};

/**
 * What is indexed, and retrieval without generation.
 *
 * Searching here is fast and shows the scores, which is how a thin answer gets
 * diagnosed: whether the model was given nothing useful, or was given good
 * chunks and made little of them.
 */
export function DocumentsPage() {
  const [indexes, setIndexes] = useState<IndexStatus[]>([]);
  const [index, setIndex] = useState("policy");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResults | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  useEffect(() => {
    getIndexes().then(setIndexes).catch(setError);
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      setResults(await search({ q: query, index, k: 8, minScore: 0 }));
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error("Search failed"));
      setResults(null);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h2>Documents</h2>
          <p className="muted">
            Retrieval without generation: fast, and it shows the scores an answer was built on.
          </p>
        </div>
      </header>

      <ErrorBanner error={error} />

      <div className="cards">
        {indexes.map((status) => (
          <div className="card" key={status.name}>
            <span className="card-label">{status.name}</span>
            <span className="card-value">{status.built ? status.vector_count : "—"}</span>
            <span className="card-note">
              {status.built
                ? `${status.dimension}-dim · ${status.embedding_model ?? "unknown model"}`
                : BUILD_COMMAND[status.name]}
            </span>
          </div>
        ))}
      </div>

      <form className="search-form" onSubmit={submit}>
        <select value={index} onChange={(event) => setIndex(event.target.value)}>
          {["policy", "company", "news"].map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search the indexed chunks…"
        />
        <button type="submit" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>

      {results ? (
        results.results.length === 0 ? (
          <Empty title="Nothing matched">{results.reason}</Empty>
        ) : (
          <ol className="hits">
            {results.results.map((hit, position) => (
              <li key={position} className="hit">
                <div className="hit-head">
                  <span className="source-score">{hit.score.toFixed(3)}</span>
                  <span className="hit-citation">{hit.citation}</span>
                </div>
                <p className="hit-text">{hit.text}</p>
                {hit.metadata.url ? (
                  <a href={hit.metadata.url} target="_blank" rel="noreferrer" className="source-link">
                    {hit.metadata.url}
                  </a>
                ) : null}
              </li>
            ))}
          </ol>
        )
      ) : null}
    </div>
  );
}
