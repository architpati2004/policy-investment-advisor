import type { ApiError } from "../services/client";

/**
 * A real failure: the server is down, a model is missing, an index is unbuilt.
 *
 * Distinct from a refusal, which is a result. Every backend error carries a
 * remediation string — the command that fixes it — and showing that is the
 * difference between a user who can act and one who files a bug.
 */
export function ErrorBanner({ error }: { error: ApiError | Error | null }) {
  if (!error) return null;
  const remediation = "remediation" in error ? (error as ApiError).remediation : null;
  return (
    <div className="banner banner-error" role="alert">
      <strong>{error.message}</strong>
      {remediation ? <div className="banner-fix">Fix: {remediation}</div> : null}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <div className="muted">{children}</div>
    </div>
  );
}
