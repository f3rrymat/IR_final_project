import { compact, fmt, methodLabel, ms, parseRoute, pct, relChange, stopSentence } from "./format";

describe("format helpers", () => {
  it("formats numbers and missing values", () => {
    expect(fmt(0.71467, 3)).toBe("0.715");
    expect(fmt(null)).toBe("—");
    expect(fmt(Number.NaN)).toBe("—");
    expect(pct(0.3612)).toBe("36%");
    expect(pct(0.0667, 1)).toBe("6.7%");
    expect(compact(1284)).toBe("1,284");
    expect(compact(12900)).toBe("12.9K");
  });

  it("formats latency on a human scale", () => {
    expect(ms(0.61)).toBe("0.6 ms");
    expect(ms(150.2)).toBe("150 ms");
    expect(ms(20852)).toBe("20.9 s");
  });

  it("computes signed relative change", () => {
    expect(relChange(25.9, 16.6)).toBeCloseTo(-0.359, 3);
    expect(relChange(0, 5)).toBe(0);
  });

  it("labels methods", () => {
    expect(methodLabel("bm25_champion")).toBe("BM25 · Champion + IE");
    expect(methodLabel("something_new")).toBe("something_new");
  });

  it("explains every stop reason the controller emits", () => {
    expect(stopSentence("confident (0.79 >= 0.70)", "auto")).toMatch(/good enough/);
    expect(stopSentence("ladder exhausted or budget", "auto")).toMatch(/no affordable step/);
    expect(stopSentence("budget: no remaining step affordable", "auto")).toMatch(/no affordable step/);
    expect(stopSentence("voc: best expected gain per cost < lambda=0.01", "auto")).toMatch(/worth its cost/);
    expect(stopSentence("voc: all steps used", "auto")).toMatch(/all of them ran/);
    expect(stopSentence("routing only (no escalation)", "route")).toMatch(/Routing only/);
    expect(stopSentence("", "bm25")).toMatch(/Fixed method/);
  });

  it("parses the router note", () => {
    expect(parseRoute("class=semantic; question form, long query; ladder=['bm25', 'expansion']")).toEqual({
      cls: "semantic",
      why: "question form, long query",
    });
  });
});
