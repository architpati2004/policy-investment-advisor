import { Fragment } from "react";

/**
 * Renders an answer with its [n] markers turned into links to the sources.
 *
 * A citation the reader cannot follow is decoration. Clicking a marker scrolls
 * its source into view, which is the difference between an answer that claims
 * grounding and one a reader can check.
 */

const MARKER = /(\[\d{1,2}\])/g;

export function AnswerBody({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  const parts = text.split(MARKER);
  return (
    <p className="answer-body">
      {parts.map((part, index) => {
        const match = /^\[(\d{1,2})\]$/.exec(part);
        if (!match) return <Fragment key={index}>{part}</Fragment>;
        const number = Number(match[1]);
        return (
          <a
            key={index}
            href={`#source-${number}`}
            className="citation-marker"
            onClick={() => onCite?.(number)}
          >
            {part}
          </a>
        );
      })}
    </p>
  );
}
