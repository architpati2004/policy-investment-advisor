/** Builders for the API shapes, so a test states only what it is about. */
import type { Answer, Assessment, Source } from "../types";

export function source(overrides: Partial<Source> = {}): Source {
  return {
    number: 1,
    citation: "RBI — deposit directions (p. 8)",
    score: 0.513,
    cited: true,
    metadata: { source: "RBI", page: 8, origin: "policy" },
    ...overrides,
  };
}

export function answer(overrides: Partial<Answer> = {}): Answer {
  return {
    question: "can a bank pay interest on a current account?",
    answer: "No interest shall be paid on deposits held in current accounts [1].",
    covered: true,
    grounded: true,
    reason: null,
    sources: [source()],
    invalid_citations: [],
    elapsed_seconds: 36.2,
    ...overrides,
  };
}

export function assessment(overrides: Partial<Assessment> = {}): Assessment {
  return {
    ...answer(),
    portfolio: "Demo Portfolio",
    holdings: ["GODREJCP"],
    affected: [],
    unaffected: ["GODREJCP"],
    impact_declared: true,
    context: { policy: 3, company: 2 },
    ...overrides,
  };
}
