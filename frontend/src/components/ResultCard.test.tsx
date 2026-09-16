import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResultCard } from "./ResultCard";
import { answer, assessment, source } from "../test/factories";

/**
 * The refusal case is the one this component exists to get right, so it is
 * tested hardest: `covered: false` is a result, not a failure, and the reason
 * usually contains the finding.
 */
describe("ResultCard", () => {
  it("shows a grounded answer with its sources and scores", () => {
    render(<ResultCard result={answer()} />);

    expect(screen.getByText(/No interest shall be paid/)).toBeInTheDocument();
    expect(screen.getByText("0.513")).toBeInTheDocument();
    expect(screen.getByText(/1 cited of 1 retrieved/)).toBeInTheDocument();
  });

  it("renders a refusal as a finding, not an error", () => {
    const refusal = answer({
      covered: false,
      grounded: false,
      answer: "the sources govern commercial banks, not NBFCs",
      reason: "the sources govern commercial banks, not NBFCs",
      sources: [],
    });

    const { container } = render(<ResultCard result={refusal} />);

    expect(screen.getByText(/Not covered by the indexed sources/)).toBeInTheDocument();
    // The finding itself must be visible — it is usually the useful part.
    expect(screen.getByText(/govern commercial banks/)).toBeInTheDocument();
    // Nothing went wrong, so it must not be dressed as an error.
    expect(container.querySelector(".result-finding")).toBeTruthy();
    expect(container.querySelector(".banner-error")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/try again|retry|failed/i)).toBeNull();
  });

  it("warns when an answer cites nothing", () => {
    render(<ResultCard result={answer({ grounded: false, sources: [source({ cited: false })] })} />);
    expect(screen.getByText(/cites no source/)).toBeInTheDocument();
  });

  it("warns about a citation that was never supplied", () => {
    render(<ResultCard result={answer({ invalid_citations: [7] })} />);
    expect(screen.getByText(/\[7\], which was never/)).toBeInTheDocument();
  });

  it("marks retrieved sources the answer ignored", () => {
    render(<ResultCard result={answer({ sources: [source(), source({ number: 2, cited: false })] })} />);
    expect(screen.getByText("not cited")).toBeInTheDocument();
    expect(screen.getByText(/1 cited of 2 retrieved/)).toBeInTheDocument();
  });

  it("reports an undeclared impact as unknown, never as nothing affected", () => {
    render(<ResultCard result={assessment({ impact_declared: false, unaffected: [] })} />);

    expect(screen.getByText(/Impact not stated/)).toBeInTheDocument();
    expect(screen.queryByText(/No holding is affected/)).toBeNull();
  });

  it("says plainly when no holding is affected", () => {
    render(<ResultCard result={assessment({ affected: [], impact_declared: true })} />);
    expect(screen.getByText(/No holding is affected according to the sources/)).toBeInTheDocument();
  });

  it("names affected holdings", () => {
    render(
      <ResultCard result={assessment({ affected: ["HDFCBANK"], unaffected: ["GODREJCP"] })} />,
    );
    expect(screen.getByText("HDFCBANK")).toBeInTheDocument();
    expect(screen.getByText(/Unaffected: GODREJCP/)).toBeInTheDocument();
  });

  it("turns citation markers into links to their source", () => {
    const { container } = render(<ResultCard result={answer()} />);
    const marker = container.querySelector("a.citation-marker");
    expect(marker).toHaveAttribute("href", "#source-1");
  });
});
