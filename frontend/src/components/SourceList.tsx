import type { Source } from "../types";

/**
 * Retrieved sources with their scores.
 *
 * Shows everything retrieved, not just what the answer cited, and marks the
 * difference. A source the model was given and ignored is evidence about the
 * answer: five retrieved and one cited is a narrow answer, five retrieved and
 * none cited is an answer with nothing behind it.
 *
 * Scores are cosine similarity. They are shown because a 0.38 match and a 0.68
 * match read identically in prose and mean very different things.
 */

const ORIGIN_LABEL: Record<string, string> = {
  policy: "POLICY",
  holding: "FILING",
  news: "NEWS",
};

export function SourceList({ sources, highlight }: { sources: Source[]; highlight?: number }) {
  if (sources.length === 0) return null;

  const cited = sources.filter((source) => source.cited).length;

  return (
    <section className="sources">
      <h4 className="sources-heading">
        Sources
        <span className="sources-count">
          {cited} cited of {sources.length} retrieved
        </span>
      </h4>
      <ol className="source-list">
        {sources.map((source) => {
          const origin = source.metadata.origin ?? "policy";
          return (
            <li
              key={source.number}
              id={`source-${source.number}`}
              className={[
                "source",
                source.cited ? "source-cited" : "source-uncited",
                highlight === source.number ? "source-highlight" : "",
              ].join(" ")}
            >
              <div className="source-head">
                <span className="source-number">[{source.number}]</span>
                <span className={`badge badge-${origin}`}>{ORIGIN_LABEL[origin] ?? "SOURCE"}</span>
                <span className="source-score" title="cosine similarity to the question">
                  {source.score.toFixed(3)}
                </span>
                {source.cited ? null : <span className="source-note">not cited</span>}
              </div>
              <div className="source-citation">{source.citation}</div>
              {source.metadata.url ? (
                <a className="source-link" href={source.metadata.url} target="_blank" rel="noreferrer">
                  {source.metadata.url}
                </a>
              ) : null}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
