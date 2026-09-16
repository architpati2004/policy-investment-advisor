import { useEffect, useState } from "react";
import { ErrorBanner, Empty } from "../components/Banner";
import { GenerationProgress } from "../components/GenerationProgress";
import { acknowledgeAlert, getAlerts, runAlerts } from "../services/alerts";
import { getPortfolio } from "../services/portfolio";
import { ApiError } from "../services/client";
import type { Alert, AlertRun } from "../types";

/**
 * Alerts, and the run that produces them.
 *
 * A run costs one local generation per holding, so the button says so before it
 * is pressed, and quiet holdings are listed afterwards. "We checked and found
 * nothing" and "we did not check" look identical unless the quiet ones are shown.
 */
export function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [run, setRun] = useState<AlertRun | null>(null);
  const [running, setRunning] = useState(false);
  const [holdings, setHoldings] = useState(0);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [showAcknowledged, setShowAcknowledged] = useState(false);

  function refresh(includeAcknowledged = showAcknowledged) {
    getAlerts(includeAcknowledged).then(setAlerts).catch(setError);
  }

  useEffect(() => {
    refresh();
    getPortfolio().then((portfolio) => setHoldings(portfolio.holdings.length)).catch(() => setHoldings(0));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAcknowledged]);

  async function check() {
    setRunning(true);
    setError(null);
    setRun(null);
    try {
      setRun(await runAlerts());
      refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error("The run failed"));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h2>Alerts</h2>
          <p className="muted">
            Raised only where the sources show a development and the model cites one. Filings and
            uncited judgements never raise an alert.
          </p>
        </div>
        <button onClick={check} disabled={running || holdings === 0}>
          {running ? "Checking…" : `Check ${holdings} holding${holdings === 1 ? "" : "s"}`}
        </button>
      </header>

      <ErrorBanner error={error} />

      {running ? (
        <>
          <p className="muted">
            One local generation per holding — roughly {holdings * 30}-{holdings * 90} seconds.
          </p>
          <GenerationProgress />
        </>
      ) : null}

      {run ? (
        <div className="banner banner-info">
          <strong>
            {run.raised} alert{run.raised === 1 ? "" : "s"} from {run.checked} holding
            {run.checked === 1 ? "" : "s"} in {run.elapsed_seconds.toFixed(0)}s
          </strong>
          <ul className="plain-list">
            {run.findings
              .filter((finding) => !finding.raised)
              .map((finding) => (
                <li key={finding.ticker} className="muted">
                  {finding.ticker}: {finding.reason ?? "nothing material"}
                </li>
              ))}
          </ul>
        </div>
      ) : null}

      <label className="toggle">
        <input
          type="checkbox"
          checked={showAcknowledged}
          onChange={(event) => setShowAcknowledged(event.target.checked)}
        />
        Include acknowledged
      </label>

      {alerts.length === 0 ? (
        <Empty title="No outstanding alerts">
          On a corpus this size that is the expected result, not a failure.
        </Empty>
      ) : (
        <ul className="alert-list">
          {alerts.map((alert) => (
            <li key={alert.id} className={alert.acknowledged ? "alert acknowledged" : "alert"}>
              <div className="alert-head">
                <span className="chip chip-affected">{alert.ticker ?? "portfolio"}</span>
                {alert.created_at ? (
                  <span className="muted">{new Date(alert.created_at).toLocaleString()}</span>
                ) : null}
                {alert.acknowledged ? (
                  <span className="muted">acknowledged</span>
                ) : (
                  <button
                    className="link-button"
                    onClick={() => acknowledgeAlert(alert.id).then(() => refresh())}
                  >
                    acknowledge
                  </button>
                )}
              </div>
              <p>{alert.summary}</p>
              <ul className="citations">
                {alert.citations.map((citation, index) => (
                  <li key={index}>
                    <span className={`badge badge-${citation.origin}`}>{citation.origin}</span>
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        {citation.citation}
                      </a>
                    ) : (
                      citation.citation
                    )}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
