import { useEffect, useRef, useState } from "react";

/**
 * The loading state for a local generation.
 *
 * A spinner is wrong here. Generation takes 30-90 seconds on this hardware, and
 * a spinner that has looked identical for forty seconds is indistinguishable
 * from a hung request — the user's only options are to wait blindly or reload,
 * and reloading throws the work away.
 *
 * So: a counter that visibly ticks, which is proof of life; and stage text that
 * tells the truth about what is happening and how long it should take. The bar
 * is deliberately indeterminate — a percentage would have to be invented, and an
 * invented percentage that stalls at 80% is worse than no percentage at all.
 */

interface Stage {
  after: number;
  label: string;
  detail?: string;
}

const STAGES: Stage[] = [
  { after: 0, label: "Searching the indexes", detail: "Embedding the question and retrieving chunks." },
  {
    after: 3,
    label: "Reading the sources",
    detail: "The model is working through the retrieved text locally.",
  },
  {
    after: 45,
    label: "Still generating",
    detail: "Local inference on this machine usually takes 30-90 seconds. Nothing is stuck.",
  },
  {
    after: 100,
    label: "Taking longer than usual",
    detail:
      "The first call of a session also loads the model into RAM. If this is the first question since starting Ollama, expect up to two minutes.",
  },
];

function stageFor(seconds: number): Stage {
  return STAGES.reduce((current, stage) => (seconds >= stage.after ? stage : current), STAGES[0]);
}

export function GenerationProgress({ model }: { model?: string }) {
  const [seconds, setSeconds] = useState(0);
  const started = useRef(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - started.current) / 1000));
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  const stage = stageFor(seconds);

  return (
    <div className="progress" role="status" aria-live="polite">
      <div className="progress-head">
        <span className="progress-label">{stage.label}</span>
        <span className="progress-elapsed" aria-label={`${seconds} seconds elapsed`}>
          {seconds}s
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-bar" />
      </div>
      <p className="progress-detail">
        {stage.detail}
        {model ? ` Running ${model}.` : ""}
      </p>
    </div>
  );
}
