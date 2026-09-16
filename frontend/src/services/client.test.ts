import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

/** The remediation string is the difference between a user who can act and one who cannot. */
describe("api client", () => {
  afterEach(() => vi.unstubAllGlobals());

  function respond(status: number, body: unknown) {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: status < 400,
        status,
        json: async () => body,
      }),
    );
  }

  it("returns the parsed body on success", async () => {
    respond(200, { name: "Demo Portfolio" });
    await expect(api.get<{ name: string }>("/api/portfolio")).resolves.toEqual({
      name: "Demo Portfolio",
    });
  });

  it("carries the backend's remediation through", async () => {
    respond(404, {
      error: "UnknownTickerError",
      message: "'INFY' is not a known company",
      remediation: "Declare it in data/companies/registry.json, then run sync",
    });

    await expect(api.post("/api/portfolio/holdings", {})).rejects.toMatchObject({
      status: 404,
      kind: "UnknownTickerError",
      message: "'INFY' is not a known company",
      remediation: expect.stringContaining("registry.json"),
    });
  });

  it("explains an unreachable server rather than throwing a bare network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(api.get("/health")).rejects.toThrow(/Is the server running/);
  });

  it("handles FastAPI validation errors, which use a different shape", async () => {
    respond(422, { detail: "question must not be empty" });
    const error = (await api.post("/api/chat/policy", {}).catch((e) => e)) as ApiError;
    expect(error.message).toBe("question must not be empty");
  });
});
