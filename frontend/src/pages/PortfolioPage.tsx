import { useEffect, useState } from "react";
import { ErrorBanner, Empty } from "../components/Banner";
import { addHolding, getCompanies, getPortfolio, getScope, removeHolding } from "../services/portfolio";
import { ApiError } from "../services/client";
import type { Company, Portfolio, Scope } from "../types";

/** Holdings, and the retrieval scope they produce. */
export function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [scope, setScope] = useState<Scope | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [form, setForm] = useState({ ticker: "", quantity: "", average_cost: "" });
  const [saving, setSaving] = useState(false);

  function reload() {
    getPortfolio().then(setPortfolio).catch(setError);
    getScope().then(setScope).catch(() => setScope(null));
  }

  useEffect(() => {
    reload();
    getCompanies().then(setCompanies).catch(() => setCompanies([]));
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      setPortfolio(await addHolding(form));
      setForm({ ticker: "", quantity: "", average_cost: "" });
      getScope().then(setScope).catch(() => setScope(null));
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error("Could not add the holding"));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h2>Portfolio</h2>
          <p className="muted">
            Cost basis only. Buying more of something held re-weights its average cost rather than
            adding a second row.
          </p>
        </div>
      </header>

      <ErrorBanner error={error} />

      <form className="holding-form" onSubmit={submit}>
        <select
          value={form.ticker}
          onChange={(event) => setForm({ ...form, ticker: event.target.value })}
          required
        >
          <option value="">Company…</option>
          {companies.map((company) => (
            <option key={company.ticker} value={company.ticker}>
              {company.ticker} — {company.name} ({company.sector})
            </option>
          ))}
        </select>
        <input
          value={form.quantity}
          onChange={(event) => setForm({ ...form, quantity: event.target.value })}
          placeholder="Shares"
          inputMode="decimal"
          required
        />
        <input
          value={form.average_cost}
          onChange={(event) => setForm({ ...form, average_cost: event.target.value })}
          placeholder="Price paid per share (₹)"
          inputMode="decimal"
          required
        />
        <button type="submit" disabled={saving}>
          {saving ? "Adding…" : "Add"}
        </button>
      </form>
      {companies.length === 0 ? (
        <p className="muted">
          No companies declared. Add them to <code>data/companies/registry.json</code> — a holding
          with no sector could never be matched to a policy change.
        </p>
      ) : null}

      {portfolio && portfolio.holdings.length > 0 ? (
        <table className="table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Sector</th>
              <th className="right">Quantity</th>
              <th className="right">Avg cost</th>
              <th className="right">Invested</th>
              <th className="right">Weight</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {portfolio.holdings.map((holding) => (
              <tr key={holding.ticker}>
                <td>
                  <strong>{holding.ticker}</strong>
                  <div className="muted">{holding.name}</div>
                </td>
                <td>{holding.sector}</td>
                <td className="right">{holding.quantity}</td>
                <td className="right">₹{holding.average_cost}</td>
                <td className="right">₹{holding.invested}</td>
                <td className="right">{holding.weight_percent}%</td>
                <td className="right">
                  <button
                    className="link-button"
                    onClick={() => removeHolding(holding.ticker).then(setPortfolio).catch(setError)}
                  >
                    remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={4}>Total at cost</td>
              <td className="right">₹{portfolio.invested}</td>
              <td colSpan={2} />
            </tr>
          </tfoot>
        </table>
      ) : (
        <Empty title="No holdings yet">
          Portfolio questions and alerts have nothing to work against until you add one.
        </Empty>
      )}

      {scope && (scope.tickers.length > 0 || scope.sectors.length > 0) ? (
        <section>
          <h3>Retrieval scope</h3>
          <p className="muted">
            What company and news searches are narrowed to. Sector matters independently: a rule
            aimed at an industry reaches a holding it never names.
          </p>
          <div className="chips">
            {scope.tickers.map((ticker) => (
              <span key={ticker} className="chip">
                {ticker}
              </span>
            ))}
            {scope.sectors.map((sector) => (
              <span key={sector} className="chip chip-sector">
                {sector}
              </span>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
