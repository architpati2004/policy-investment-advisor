/**
 * The one place that talks to the network.
 *
 * Application errors arrive as `{error, message, remediation}` — every backend
 * exception carries the command that fixes it — so those are surfaced intact
 * rather than flattened into "request failed". A user told "run: ollama pull
 * qwen3:1.7b" can act; a user told "500" cannot.
 */

export interface ApiErrorBody {
  error?: string;
  message?: string;
  remediation?: string | null;
  detail?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly remediation: string | null;
  readonly kind: string;

  constructor(status: number, body: ApiErrorBody | null, fallback: string) {
    super(body?.message ?? fallback);
    this.name = "ApiError";
    this.status = status;
    this.kind = body?.error ?? "HTTPError";
    this.remediation = body?.remediation ?? null;
  }
}

/** No timeout: a generation legitimately takes 30-90 seconds on local hardware. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(0, null, "Cannot reach the API. Is the server running on port 8000?");
  }

  if (!response.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      body = null;
    }
    // FastAPI validation errors use `detail` rather than our error shape.
    if (body && !body.message && body.detail) {
      body.message = typeof body.detail === "string" ? body.detail : "The request was rejected.";
    }
    throw new ApiError(response.status, body, `Request failed (${response.status})`);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
