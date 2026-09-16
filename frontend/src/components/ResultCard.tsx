import { useState } from "react";
import { AnswerBody } from "./AnswerBody";
import { SourceList } from "./SourceList";
import type { Answer, Assessment } from "../types";

/**
 * An answer, or a refusal shown as a finding.
 *
 * The refusal case is the one worth getting right. `covered: false` means the
 * indexed documents do not answer the question — and the model's explanation of
 * *why* routinely contains the useful part: "the sources govern commercial
 * banks, so this does not reach an FMCG holding" is a finding, not a failure.
 * The two models even disagree about the label while agreeing on the content.
 *
 * So a refusal is rendered as a neutral, informative card — never red, never an
 * error icon, never a retry prompt. Nothing went wrong. What it does not do is
 * dress a refusal up as an answer: the heading says plainly that the corpus does
 * not cover this, so the reader knows the standing of what follows.
 */

function isAssessment(result: Answer | Assessment): result is Assessment {
  return "impact_declared" in result;
}

export function ResultCard({ result }: { result: Answer | Assessment }) {
  const [highlight, setHighlight] = useState<number | undefined>();

  return (
    <article className={`result ${result.covered ? "result-answer" : "result-finding"}`}>
      {result.covered ? null : (
        <header className="finding-head">
          <span className="finding-tag">Not covered by the indexed sources</span>
          <p className="finding-lead">
            The documents in the index do not answer this. What the model found is below — it is
            often the useful part.
          </p>
        </header>
      )}

      <AnswerBody
        text={result.covered ? result.answer : result.reason ?? result.answer}
        onCite={setHighlight}
      />

      {isAssessment(result) ? <ImpactSummary assessment={result} /> : null}

      {result.covered && !result.grounded ? (
        <p className="caution">
          This answer cites no source, so nothing in the index backs it. Treat it as unverified.
        </p>
      ) : null}

      {result.invalid_citations.length > 0 ? (
        <p className="caution">
          It cited {result.invalid_citations.map((n) => `[${n}]`).join(", ")}, which was never
          supplied. Those references point at nothing.
        </p>
      ) : null}

      <SourceList sources={result.sources} highlight={highlight} />

      <footer className="result-meta">
        {result.elapsed_seconds.toFixed(1)}s
        {isAssessment(result) && result.context
          ? ` · ${result.context.policy ?? 0} policy + ${result.context.company ?? 0} company chunks`
          : null}
      </footer>
    </article>
  );
}

function ImpactSummary({ assessment }: { assessment: Assessment }) {
  if (!assessment.covered) return null;

  // Undeclared impact is *unknown*, not "nothing affected". Presenting silence
  // as safety is the one reading this must never allow.
  if (!assessment.impact_declared) {
    return (
      <div className="impact impact-unknown">
        <strong>Impact not stated.</strong> The assessment did not say which holdings are affected,
        so this is unknown rather than none.
      </div>
    );
  }

  return (
    <div className={`impact ${assessment.affected.length ? "impact-affected" : "impact-clear"}`}>
      {assessment.affected.length ? (
        <>
          <strong>Affected:</strong>{" "}
          {assessment.affected.map((ticker) => (
            <span key={ticker} className="chip chip-affected">
              {ticker}
            </span>
          ))}
        </>
      ) : (
        <strong>No holding is affected according to the sources.</strong>
      )}
      {assessment.unaffected.length ? (
        <div className="impact-rest">
          Unaffected: {assessment.unaffected.join(", ")}
        </div>
      ) : null}
    </div>
  );
}
