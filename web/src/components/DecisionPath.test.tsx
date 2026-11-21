import { render, screen, within } from "@testing-library/react";
import type { SearchResponse } from "../lib/api";
import auto from "../test/fixtures/search_auto.json";
import bm25 from "../test/fixtures/search_bm25.json";
import skips from "../test/fixtures/search_skips.json";
import stops from "../test/fixtures/search_auto_stops.json";
import { buildPath, DecisionPath, spreadPositions, SpendLedger } from "./DecisionPath";

const r = (x: unknown) => x as SearchResponse;

describe("buildPath", () => {
  it("turns a full escalation into route → retrieve/confidence pairs → stop", () => {
    const p = buildPath(r(auto).trace, r(auto).tau, "auto");
    expect(p[0]).toMatchObject({ kind: "route", method: "bm25_champion", cls: "default" });
    expect(p.filter((n) => n.kind === "retrieve").map((n) => (n as { method: string }).method)).toEqual(r(auto).methods_run);
    const conf = p.filter((n) => n.kind === "confidence");
    expect(conf).toHaveLength(r(auto).methods_run.length);
    expect(conf.every((n) => n.kind === "confidence" && n.passed === false)).toBe(true); // tau 0.99 never reached
    expect(p.at(-1)).toMatchObject({ kind: "stop", reason: "ladder exhausted or budget" });
  });

  it("marks a confident stop as passed", () => {
    const p = buildPath(r(stops).trace, r(stops).tau, "auto");
    expect(p.find((n) => n.kind === "confidence")).toMatchObject({ passed: true });
  });

  it("shows unaffordable steps as skipped", () => {
    const p = buildPath(r(skips).trace, r(skips).tau, "auto");
    expect(p.filter((n) => n.kind === "skip").map((n) => (n as { method: string }).method)).toEqual(["bm25", "dense", "expansion"]);
  });

  it("a fixed method is a single retrieve step", () => {
    expect(buildPath(r(bm25).trace, null, "bm25")).toEqual([
      expect.objectContaining({ kind: "retrieve", method: "bm25" }),
    ]);
  });
});

describe("<DecisionPath>", () => {
  it("renders every step in order with τ comparisons", () => {
    render(<DecisionPath trace={r(auto).trace} tau={r(auto).tau} mode="auto" />);
    const path = screen.getByRole("list", { name: "Decision path" });
    const items = within(path).getAllByRole("listitem").filter((li) => li.textContent);
    expect(items[0]).toHaveTextContent(/Route · default/);
    expect(items.at(-1)).toHaveTextContent(/Stop/);
    expect(within(path).getAllByText(/< τ 0\.99/).length).toBeGreaterThan(0);
    expect(within(path).getByText("Dense")).toBeInTheDocument();
  });

  it("renders skipped steps with the reason", () => {
    render(<DecisionPath trace={r(skips).trace} tau={r(skips).tau} mode="auto" />);
    expect(screen.getByText(/Dense skipped/)).toBeInTheDocument();
    expect(screen.getByText(/needs 23.30, 0.32 left/)).toBeInTheDocument();
  });
});

describe("spreadPositions", () => {
  it("keeps readings apart when one step dominates the budget", () => {
    // a cheap BM25 step followed by a dense step that spends almost everything
    const out = spreadPositions([0.3, 0.5, 52]);
    for (let i = 1; i < out.length; i++) expect(out[i] - out[i - 1]).toBeGreaterThanOrEqual(16 - 1e-9);
    expect(Math.min(...out)).toBeGreaterThanOrEqual(5);
    expect(Math.max(...out)).toBeLessThanOrEqual(95);
  });

  it("pulls readings bunched at the right edge back inside", () => {
    const out = spreadPositions([99, 99.5, 100]);
    expect(out).toEqual([63, 79, 95]);
  });

  it("fits many readings into the plot", () => {
    const out = spreadPositions([100, 100, 100, 100, 100, 100, 100]);
    expect(out[0]).toBeCloseTo(5);
    expect(out.at(-1)).toBeCloseTo(95);
  });

  it("leaves well separated readings where they are", () => {
    expect(spreadPositions([10, 50, 90])).toEqual([10, 50, 90]);
  });
});

describe("<SpendLedger>", () => {
  it("labels every confidence reading with its value", () => {
    const a = r(auto);
    render(<SpendLedger trace={a.trace} tau={a.tau} budget={a.budget} mode="auto" total={a.total_cost} />);
    const ledger = screen.getByRole("group", { name: "Budget ledger" });
    const values = a.trace.filter((t) => t.action === "confidence").map((t) => (t.confidence as number).toFixed(2));
    for (const v of values) expect(within(ledger).getAllByText(v).length).toBeGreaterThan(0);
  });
});
