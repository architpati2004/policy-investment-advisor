import { api } from "./client";
import type { Answer, Assessment } from "../types";

/** Ask the policy corpus. Expect 30-90 seconds: this runs a local model. */
export function askPolicy(question: string): Promise<Answer> {
  return api.post<Answer>("/api/chat/policy", { question });
}

/** Assess a question against the portfolio's holdings. Same latency. */
export function assessPortfolio(question: string, portfolio?: string): Promise<Assessment> {
  return api.post<Assessment>("/api/chat/portfolio", { question, portfolio });
}
