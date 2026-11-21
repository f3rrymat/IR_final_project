// Small, pure formatting helpers (unit-tested in format.test.ts).

export const METHOD_LABEL: Record<string, string> = {
  auto: "Adaptive",
  bm25: "BM25",
  tfidf: "TF-IDF",
  dense: "Dense",
  hybrid: "Fixed hybrid",
  boolean: "Boolean",
  bm25_champion: "BM25 · Champion + IE",
  bm25_ie: "BM25 · Index elimination",
  route: "Routing only",
  expansion: "Query expansion",
  fused: "Fused",
  fusion: "Fused",
};

export const methodLabel = (m: string | null | undefined) => (m ? METHOD_LABEL[m] ?? m : "");

/** Fixed decimals, "—" for null/NaN. */
export function fmt(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

/** 1284 -> "1,284"; 12900 -> "12.9K". */
export function compact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 10000) return `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}K`;
  return n.toLocaleString("en-US");
}

/** Latency: sub-millisecond as "0.6 ms", seconds past 1000 ms. */
export function ms(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (x >= 1000) return `${(x / 1000).toFixed(1)} s`;
  if (x >= 100) return `${Math.round(x)} ms`;
  return `${x.toFixed(1)} ms`;
}

/** Relative change b vs a, signed: (b - a) / a. */
export function relChange(a: number, b: number): number {
  return a === 0 ? 0 : (b - a) / a;
}

export const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Plain-language reading of the controller's stop reason. */
export function stopSentence(reason: string, mode: string): string {
  if (reason.startsWith("routing only")) return "Routing only: one method, chosen from the query, no escalation.";
  if (mode !== "auto") return "Fixed method: no confidence check, nothing to escalate.";
  if (!reason) return "";
  if (reason.startsWith("confident")) return "Stopped early: the results already looked good enough.";
  if (reason.startsWith("voc: best expected gain")) return "Stopped: no remaining step was worth its cost.";
  if (reason.startsWith("voc: all steps")) return "Every step was worth trying, so all of them ran.";
  if (reason.includes("budget") || reason.includes("exhausted"))
    return "Stopped: confidence stayed below τ and no affordable step was left.";
  return reason;
}

/** "class=semantic; question form, long query; ladder=[...]" -> { cls, why } */
export function parseRoute(note: string): { cls: string; why: string } {
  const parts = note.split(";").map((s) => s.trim());
  const cls = (parts.find((p) => p.startsWith("class="))?.slice(6) ?? "").replace(/_/g, " ");
  const why = parts.filter((p) => !p.startsWith("class=") && !p.startsWith("ladder=")).join("; ");
  return { cls, why };
}

export type MethodKind = "lexical" | "semantic" | "expand" | "fused";

/** Which family a method belongs to; drives its colour everywhere (lexical cobalt, semantic violet, expansion amber). */
export function methodKind(m: string | null | undefined): MethodKind {
  if (m === "dense") return "semantic";
  if (m === "expansion") return "expand";
  if (m === "fusion" || m === "fused" || m === "hybrid" || m === "auto") return "fused";
  return "lexical";
}

/** CSS colour for a method family, for SVG fills and inline styles. */
export const KIND_COLOR: Record<MethodKind, string> = {
  lexical: "rgb(var(--lexical))",
  semantic: "rgb(var(--semantic))",
  expand: "rgb(var(--expand))",
  fused: "rgb(var(--ink))",
};

export const signed = (x: number | null | undefined, digits = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
