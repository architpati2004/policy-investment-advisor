import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GenerationProgress } from "./GenerationProgress";

/**
 * A 36-second wait needs to prove it is alive. These tests pin the two things
 * that make it do so: a counter that advances, and stage text that changes as
 * the wait gets long enough to look broken.
 */
describe("GenerationProgress", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("counts up so the wait never looks frozen", () => {
    render(<GenerationProgress />);
    expect(screen.getByText("0s")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(12_000);
    });

    expect(screen.getByText("12s")).toBeInTheDocument();
  });

  it("explains a long wait rather than leaving it unexplained", () => {
    render(<GenerationProgress />);
    expect(screen.getByText(/Searching the indexes/)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(50_000);
    });

    expect(screen.getByText(/Still generating/)).toBeInTheDocument();
    expect(screen.getByText(/Nothing is stuck/)).toBeInTheDocument();
  });

  it("mentions the model load once the wait is unusual", () => {
    render(<GenerationProgress />);
    act(() => {
      vi.advanceTimersByTime(110_000);
    });
    expect(screen.getByText(/loads the model into RAM/)).toBeInTheDocument();
  });

  it("names the model so the cost is attributable", () => {
    render(<GenerationProgress model="qwen3:1.7b" />);
    expect(screen.getByText(/qwen3:1.7b/)).toBeInTheDocument();
  });

  it("announces itself to assistive technology", () => {
    render(<GenerationProgress />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});
