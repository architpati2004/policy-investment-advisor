import { api } from "./client";
import type { Company, Portfolio, Scope } from "../types";

export function getPortfolio(name?: string): Promise<Portfolio> {
  return api.get<Portfolio>(`/api/portfolio${name ? `?name=${encodeURIComponent(name)}` : ""}`);
}

export function getScope(): Promise<Scope> {
  return api.get<Scope>("/api/portfolio/scope");
}

export function getCompanies(): Promise<Company[]> {
  return api.get<Company[]>("/api/portfolio/companies");
}

export function addHolding(input: {
  ticker: string;
  quantity: string;
  average_cost: string;
  bought_on?: string | null;
}): Promise<Portfolio> {
  return api.post<Portfolio>("/api/portfolio/holdings", input);
}

export function removeHolding(ticker: string): Promise<Portfolio> {
  return api.delete<Portfolio>(`/api/portfolio/holdings/${encodeURIComponent(ticker)}`);
}
