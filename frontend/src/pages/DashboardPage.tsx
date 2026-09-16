import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ErrorBanner, Empty } from "../components/Banner";
import { getPortfolio } from "../services/portfolio";
import { getAlerts } from "../services/alerts";
import { getIndexes } from "../services/documents";
import type { Alert, IndexStatus, Portfolio } from "../types";
import { ApiError } from "../services/client";

/** What is in the system right now, and what it is missing. */
export function DashboardPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [indexes, setIndexes] = useState<IndexStatus[]>([]);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    getPortfolio().then(setPortfolio).catch(setError);
    getAlerts().then(setAlerts).catch(() => setAlerts([]));
    getIndexes().then(setIndexes).catch(() => setIndexes([]));
  }, []);

  const unbuilt = indexes.filter((index) => !index.built);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h2>Dashboard</h2>
          <p className="muted">Cost basis only — there is no price feed, so nothing here is a valuation.</p>
        </div>
      </header>

      <ErrorBanner error={error} />

      <div className="cards">
        <div className="card">
          <span className="card-label">Invested</span>
          <span className="card-value">{portfolio ? `₹${portfolio.invested}` : "—"}</span>
          <span className="card-note">{portfolio?.holdings.length ?? 0} holdings</span>
        </div>
        <div className="card">
          <span className="card-label">Outstanding alerts</span>
          <span className="card-value">{alerts.length}</span>
          <span className="card-note">
            <Link to="/alerts">review</Link>
          </span>
        </div>
        {indexes.map((index) => (
          <div className="card" key={index.name}>
            <span className="card-label">{index.name} index</span>
            <span className="card-value">{index.built ? index.vector_count : "—"}</span>
            <span className="card-note">{index.built ? "chunks" : "not built"}</span>
          </div>
        ))}
      </div>

      {unbuilt.length > 0 ? (
        <div className="banner banner-info">
          <strong>
            {unbuilt.map((index) => index.name).join(", ")} {unbuilt.length === 1 ? "index is" : "indexes are"} not built.
          </strong>
          <div className="banner-fix">
            Questions will find nothing there until you run the matching build script.
          </div>
        </div>
      ) : null}

      <section>
        <h3>Sector exposure</h3>
        {portfolio && portfolio.holdings.length > 0 ? (
          <div className="bars">
            {Object.entries(portfolio.sector_weights).map(([sector, weight]) => (
              <div className="bar-row" key={sector}>
                <span className="bar-label">{sector}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Number(weight)}%` }} />
                </div>
                <span className="bar-value">{weight}%</span>
              </div>
            ))}
          </div>
        ) : (
          <Empty title="No holdings yet">
            Add one on the <Link to="/portfolio">Portfolio</Link> page, and impact questions will have
            something to assess against.
          </Empty>
        )}
      </section>

      <section>
        <h3>Recent alerts</h3>
        {alerts.length === 0 ? (
          <Empty title="Nothing outstanding">
            On a small corpus that is the expected result — an engine that fires constantly is noise.
          </Empty>
        ) : (
          <ul className="plain-list">
            {alerts.slice(0, 3).map((alert) => (
              <li key={alert.id}>
                <span className="chip">{alert.ticker ?? "portfolio"}</span> {alert.summary}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
