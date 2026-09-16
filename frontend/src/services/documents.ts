import { api } from "./client";
import type { IndexStatus, SearchResults, SystemConfig } from "../types";

export function getIndexes(): Promise<IndexStatus[]> {
  return api.get<IndexStatus[]>("/api/documents/indexes");
}

export function search(params: {
  q: string;
  index: string;
  k?: number;
  minScore?: number;
  company?: string;
  sector?: string;
}): Promise<SearchResults> {
  const query = new URLSearchParams({ q: params.q, index: params.index });
  if (params.k) query.set("k", String(params.k));
  if (params.minScore !== undefined) query.set("min_score", String(params.minScore));
  if (params.company) query.set("company", params.company);
  if (params.sector) query.set("sector", params.sector);
  return api.get<SearchResults>(`/api/documents/search?${query.toString()}`);
}

export function getConfig(): Promise<SystemConfig> {
  return api.get<SystemConfig>("/api/config");
}
