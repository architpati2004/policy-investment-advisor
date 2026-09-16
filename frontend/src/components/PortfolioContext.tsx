import type { Portfolio } from "../types";

/**
 * What "my holdings" currently means, shown beside the chat.
 *
 * A portfolio question is answered against whatever is in the database, and a
 * user who has forgotten what that is cannot judge the answer. Showing the
 * holdings makes the scope of "does this affect me" visible before the question
 * is asked rather than after.
 */
export function PortfolioContext({
  portfolio,
  error,
}: {
  portfolio: Portfolio | null;
  error: string | null;
}) {
  if (error) {
    return (
      <aside className="context context-empty">
        <h3>Portfolio</h3>
        <p className="muted">{error}</p>
      </aside>
    );
  }

  if (!portfolio) {
    return (
      <aside className="context">
        <h3>Portfolio</h3>
        <p className="muted">Loading…</p>
      </aside>
    );
  }

  if (portfolio.holdings.length === 0) {
    return (
      <aside className="context context-empty">
        <h3>Portfolio</h3>
        <p className="muted">
          No holdings yet. Portfolio questions have nothing to assess against until you add one.
        </p>
      </aside>
    );
  }

  return (
    <aside className="context">
      <h3>Portfolio</h3>
      <ul className="context-holdings">
        {portfolio.holdings.map((holding) => (
          <li key={holding.ticker}>
            <span className="chip">{holding.ticker}</span>
            <span className="context-sector">{holding.sector}</span>
            <span className="context-weight">{holding.weight_percent}%</span>
          </li>
        ))}
      </ul>
      <p className="context-total">
        ₹{holding_total(portfolio)} at cost
        <span className="muted"> · cost basis, not market value</span>
      </p>
    </aside>
  );
}

/** Formatted from the string the API sent; never parsed into a float. */
function holding_total(portfolio: Portfolio): string {
  const [whole, fraction = "00"] = portfolio.invested.split(".");
  return `${Number(whole).toLocaleString("en-IN")}.${fraction}`;
}
