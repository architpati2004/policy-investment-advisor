import { useEffect, useRef, useState } from "react";
import { GenerationProgress } from "../components/GenerationProgress";
import { PortfolioContext } from "../components/PortfolioContext";
import { ResultCard } from "../components/ResultCard";
import { ErrorBanner } from "../components/Banner";
import { askPolicy, assessPortfolio } from "../services/chat";
import { getPortfolio } from "../services/portfolio";
import { getConfig } from "../services/documents";
import { ApiError } from "../services/client";
import type { Answer, Assessment, Portfolio } from "../types";

type Mode = "policy" | "portfolio";

interface Turn {
  id: number;
  question: string;
  mode: Mode;
  result: Answer | Assessment | null;
  error: ApiError | Error | null;
}

const EXAMPLES: Record<Mode, string[]> = {
  policy: [
    "Can a bank pay interest on a current account?",
    "What is the penalty for premature withdrawal of a term deposit?",
    "What changed in the cash reserve ratio?",
  ],
  portfolio: [
    "Do the new commercial bank deposit rate rules affect my holdings?",
    "What regulatory risks affect consumer goods?",
    "What do my holdings report about input costs?",
  ],
};

export function ChatPage() {
  const [mode, setMode] = useState<Mode>("portfolio");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState(false);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);
  const [model, setModel] = useState<string | undefined>();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getPortfolio()
      .then(setPortfolio)
      .catch((error: ApiError) => setPortfolioError(error.message));
    getConfig()
      .then((config) => setModel(config.llm_model))
      .catch(() => setModel(undefined));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, pending]);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;

    const id = Date.now();
    setTurns((previous) => [...previous, { id, question: trimmed, mode, result: null, error: null }]);
    setQuestion("");
    setPending(true);

    try {
      const result =
        mode === "policy" ? await askPolicy(trimmed) : await assessPortfolio(trimmed);
      setTurns((previous) =>
        previous.map((turn) => (turn.id === id ? { ...turn, result } : turn)),
      );
    } catch (caught) {
      const error = caught instanceof Error ? caught : new Error("Something went wrong");
      setTurns((previous) =>
        previous.map((turn) => (turn.id === id ? { ...turn, error } : turn)),
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="chat-layout">
      <div className="chat-main">
        <header className="page-head">
          <div>
            <h2>AI Research Chat</h2>
            <p className="muted">
              Answers are grounded in the documents you have indexed, with the sources they came
              from. When the corpus does not cover a question, it says so instead of guessing.
            </p>
          </div>
          <div className="mode-switch" role="tablist" aria-label="Question type">
            <button
              role="tab"
              aria-selected={mode === "portfolio"}
              className={mode === "portfolio" ? "mode active" : "mode"}
              onClick={() => setMode("portfolio")}
              disabled={pending}
            >
              Portfolio impact
            </button>
            <button
              role="tab"
              aria-selected={mode === "policy"}
              className={mode === "policy" ? "mode active" : "mode"}
              onClick={() => setMode("policy")}
              disabled={pending}
            >
              Policy question
            </button>
          </div>
        </header>

        <div className="transcript">
          {turns.length === 0 ? (
            <div className="examples">
              <p className="muted">
                {mode === "portfolio"
                  ? "Asked against your holdings, using policy, filings and news."
                  : "Asked against the policy index only."}
              </p>
              {EXAMPLES[mode].map((example) => (
                <button key={example} className="example" onClick={() => ask(example)} disabled={pending}>
                  {example}
                </button>
              ))}
            </div>
          ) : null}

          {turns.map((turn) => (
            <div key={turn.id} className="turn">
              <div className="question">
                <span className={`chip chip-mode chip-${turn.mode}`}>
                  {turn.mode === "portfolio" ? "portfolio" : "policy"}
                </span>
                {turn.question}
              </div>
              {turn.error ? <ErrorBanner error={turn.error} /> : null}
              {turn.result ? <ResultCard result={turn.result} /> : null}
              {!turn.result && !turn.error ? <GenerationProgress model={model} /> : null}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            ask(question);
          }}
        >
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                ask(question);
              }
            }}
            placeholder={
              mode === "portfolio"
                ? "Ask what a change means for your holdings…"
                : "Ask about the indexed regulation…"
            }
            rows={2}
            disabled={pending}
          />
          <button type="submit" disabled={pending || !question.trim()}>
            {pending ? "Generating…" : "Ask"}
          </button>
        </form>
        <p className="disclaimer">
          Grounded in the indexed documents and run entirely on this machine. Research aid, not
          investment advice.
        </p>
      </div>

      <PortfolioContext portfolio={portfolio} error={portfolioError} />
    </div>
  );
}
