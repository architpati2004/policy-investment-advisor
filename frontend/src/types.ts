/**
 * API contracts, mirroring backend/api/schemas.py.
 *
 * Money arrives as strings, not numbers: a JSON number cannot hold 1180.50
 * exactly and a portfolio sums these. Anything formatted for display is
 * formatted from the string, never parsed into a float first.
 */

export interface Holding {
  ticker: string;
  name: string;
  sector: string;
  quantity: string;
  average_cost: string;
  invested: string;
  weight_percent: string;
  first_bought_on: string | null;
}

export interface Portfolio {
  name: string;
  invested: string;
  holdings: Holding[];
  sector_weights: Record<string, string>;
}

export interface Scope {
  tickers: string[];
  sectors: string[];
}

export interface Company {
  ticker: string;
  name: string;
  sector: string;
}

export interface SourceMetadata {
  source?: string;
  title?: string;
  page?: number;
  date?: string;
  company?: string;
  sector?: string;
  url?: string;
  origin?: "policy" | "holding" | "news";
  document_type?: string;
  [key: string]: unknown;
}

export interface Source {
  number: number;
  citation: string;
  score: number;
  cited: boolean;
  metadata: SourceMetadata;
}

/**
 * An answer *or* a refusal. `covered: false` is a result, not an error — the
 * refusal text usually contains the finding, so it is displayed, never hidden.
 */
export interface Answer {
  question: string;
  answer: string;
  covered: boolean;
  grounded: boolean;
  reason: string | null;
  sources: Source[];
  invalid_citations: number[];
  elapsed_seconds: number;
}

export interface Assessment extends Answer {
  portfolio: string;
  holdings: string[];
  affected: string[];
  unaffected: string[];
  /** False means impact is *unknown*, which must never be shown as "none affected". */
  impact_declared: boolean;
  context: { policy?: number; company?: number };
}

export interface IndexStatus {
  name: string;
  built: boolean;
  vector_count: number;
  dimension: number | null;
  directory: string | null;
  embedding_model: string | null;
  updated_at: string | null;
}

export interface SearchHit {
  text: string;
  score: number;
  citation: string;
  metadata: SourceMetadata;
}

export interface SearchResults {
  query: string;
  index: string;
  results: SearchHit[];
  considered: number;
  reason: string | null;
}

export interface Citation {
  citation: string;
  origin: string;
  score: number | null;
  chunk_id: string | null;
  url: string | null;
  date: string | null;
}

export interface Alert {
  id: number;
  ticker: string | null;
  summary: string;
  citations: Citation[];
  created_at: string | null;
  acknowledged: boolean;
}

export interface Finding {
  ticker: string;
  raised: boolean;
  duplicate: boolean;
  summary: string | null;
  reason: string | null;
  considered: number;
  elapsed_seconds: number;
  citations: Citation[];
}

export interface AlertRun {
  portfolio: string;
  raised: number;
  checked: number;
  findings: Finding[];
  elapsed_seconds: number;
}

export interface SystemConfig {
  app_name: string;
  app_version: string;
  llm_model: string;
  embedding_model: string;
  retrieval_top_k: number;
  policy_index_built: boolean;
  company_index_built: boolean;
}
