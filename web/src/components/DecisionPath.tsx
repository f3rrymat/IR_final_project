import clsx from "clsx";
import { Ban, CircleCheck, CircleDot, Compass, Gauge, Search, Sparkles, Square, Wand2 } from "lucide-react";
import type { ReactNode } from "react";
import type { TraceStep } from "../lib/api";
import { fmt, KIND_COLOR, methodKind, methodLabel, ms, parseRoute } from "../lib/format";

export type PathNode =
  | { kind: "route"; method: string; cls: string; why: string; features: TraceStep["signals"] }
  | { kind: "retrieve"; method: string; cost: number; latency: number; note: string; budgetLeft: number }
  | { kind: "confidence"; value: number; tau: number | null; passed: boolean | null }
  | { kind: "skip"; method: string; note: string }
  | { kind: "stop"; reason: string };

/** Collapse the raw trace into the nodes the path draws (escalations become the links between them). */
export function buildPath(trace: TraceStep[], tau: number | null, mode: string): PathNode[] {
  const nodes: PathNode[] = [];
  for (const s of trace) {
    if (s.action === "route") {
      const { cls, why } = parseRoute(s.note);
      nodes.push({ kind: "route", method: s.method ?? "", cls, why, features: s.signals });
    } else if (s.action === "retrieve") {
      nodes.push({ kind: "retrieve", method: s.method ?? "", cost: s.cost, latency: s.latency_ms, note: s.note, budgetLeft: s.budget_left });
    } else if (s.action === "confidence" && s.confidence != null) {
      const t = mode === "auto" || mode === "route" ? tau : null;
      nodes.push({ kind: "confidence", value: s.confidence, tau: t, passed: t == null ? null : s.confidence >= t });
    } else if (s.action === "escalate" && s.note.startsWith("skip")) {
      nodes.push({ kind: "skip", method: s.method ?? "", note: s.note });
    } else if (s.action === "stop") {
      nodes.push({ kind: "stop", reason: s.note });
    }
  }
  return nodes;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* Budget ledger: the budget as a track that each method's cost fills in its own colour, with the confidence      */
/* measured after each step plotted above it against τ. One look shows where the effort went and why it stopped.  */
/* ------------------------------------------------------------------------------------------------------------ */

/** Dot positions (percent of the plot width): each sits over the step that produced it, nudged apart so a cheap
 *  step next to an expensive one never stacks two readings on top of each other. */
export function spreadPositions(xs: number[], gap = 16, pad = 5): number[] {
  const n = xs.length;
  if (!n) return [];
  const g = n > 1 ? Math.min(gap, (100 - 2 * pad) / (n - 1)) : 0;
  const out = xs.map((v) => Math.min(Math.max(v, pad), 100 - pad));
  for (let i = 1; i < n; i++) out[i] = Math.max(out[i], out[i - 1] + g);
  out[n - 1] = Math.min(out[n - 1], 100 - pad);
  for (let i = n - 2; i >= 0; i--) out[i] = Math.min(out[i], out[i + 1] - g);
  return out;
}

export function SpendLedger({
  trace,
  tau,
  budget,
  mode,
  total,
}: {
  trace: TraceStep[];
  tau: number | null;
  budget: number | null;
  mode: string;
  total: number;
}) {
  const adaptive = mode === "auto";
  const track = Math.max(adaptive && budget ? budget : 0, total, 1e-9);
  const segs: { method: string; start: number; cost: number }[] = [];
  const checks: { at: number; mid: number; value: number; method: string }[] = [];
  const skips: { method: string; need: number | null }[] = [];
  let spent = 0;
  for (const s of trace) {
    if (s.action === "retrieve") {
      segs.push({ method: s.method ?? "", start: spent, cost: s.cost });
      spent += s.cost;
    } else if (s.action === "confidence" && s.confidence != null) {
      const seg = segs[segs.length - 1];
      checks.push({ at: spent, mid: seg ? seg.start + seg.cost / 2 : spent, value: s.confidence, method: seg?.method ?? "" });
    } else if (s.action === "escalate" && s.note.startsWith("skip")) {
      const m = s.note.match(/needs ([\d.]+)/);
      skips.push({ method: s.method ?? "", need: m ? Number(m[1]) : null });
    }
  }
  const showConf = (adaptive || mode === "route") && checks.length > 0;
  const W = 100; // percentage space
  const H = 64;
  const x = (c: number) => (c / track) * W;
  // zoom the confidence axis to the values and τ, so small differences near the threshold stay visible
  const vals = [...checks.map((c) => c.value), ...(tau != null ? [tau] : [])];
  const lo = Math.max(0, Math.min(...vals, 1) - 0.08);
  const hi = Math.min(1, Math.max(...vals, 0) + 0.04);
  const y = (v: number) => 18 + (1 - (v - lo) / Math.max(hi - lo, 1e-6)) * (H - 24);
  const px = spreadPositions(checks.map((c) => x(c.mid)));
  const stepLine = checks
    .map((c, i) => `${i ? "L" : "M"}${px[i]},${y(c.value)}${checks[i + 1] ? ` L${px[i + 1]},${y(c.value)}` : ""}`)
    .join(" ");

  return (
    <div aria-label="Budget ledger" role="group">
      {showConf && (
        <div className="relative" style={{ height: H }}>
          {/* plot area leaves a gutter on the right for the τ label, so no reading can sit on top of it */}
          <div className="absolute inset-y-0 left-0 right-12">
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible" aria-hidden>
              {tau != null && (
                <line x1={0} x2={W} y1={y(tau)} y2={y(tau)} className="stroke-ink/45" strokeDasharray="1.2 1.2" vectorEffect="non-scaling-stroke" strokeWidth={1} />
              )}
              <path d={stepLine} fill="none" className="stroke-ink-faint" strokeWidth={1.25} vectorEffect="non-scaling-stroke" />
            </svg>
            {checks.map((c, i) => {
              const ok = tau != null && c.value >= tau;
              return (
                <span key={i}>
                  <span
                    title={`confidence ${fmt(c.value)} after ${methodLabel(c.method)} (${fmt(c.at, 1)} units spent)`}
                    className={clsx(
                      "absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface",
                      tau == null ? "bg-ink-faint" : ok ? "bg-pass" : "bg-fail",
                    )}
                    style={{ left: `${px[i]}%`, top: y(c.value) }}
                  />
                  <span
                    className={clsx(
                      "num absolute -translate-x-1/2 text-[0.68rem] font-medium leading-none",
                      tau == null ? "text-ink-faint" : ok ? "text-pass" : "text-fail",
                    )}
                    style={{ left: `${px[i]}%`, top: y(c.value) - 15 }}
                    aria-hidden
                  >
                    {fmt(c.value)}
                  </span>
                </span>
              );
            })}
          </div>
          {tau != null && (
            <span
              className="num absolute right-0 -translate-y-1/2 text-[0.72rem] font-medium text-ink-soft"
              style={{ top: y(tau) }}
            >
              τ {fmt(tau)}
            </span>
          )}
        </div>
      )}
      <div className="relative mt-1 h-7 overflow-hidden rounded-[7px] bg-sunken">
        {segs.map((s, i) => (
          <span
            key={i}
            title={`${methodLabel(s.method)}: ${fmt(s.cost)} units`}
            className="absolute inset-y-0 flex origin-left animate-grow items-center overflow-hidden border-r-2 border-surface px-1.5 text-[0.72rem] font-medium text-white"
            style={{
              left: `${x(s.start)}%`,
              width: `${Math.max(x(s.cost), 0.8)}%`,
              background: KIND_COLOR[methodKind(s.method)],
              animationDelay: `${i * 120}ms`,
            }}
          >
            {x(s.cost) > 14 && <span className="truncate">{methodLabel(s.method)}</span>}
          </span>
        ))}
        {adaptive &&
          skips.map((s, i) =>
            s.need ? (
              <span
                key={`k${i}`}
                title={`${methodLabel(s.method)} would need ${fmt(s.need)} units`}
                className="absolute inset-y-[3px] rounded-[5px] border border-dashed border-ink-faint/70"
                style={{ left: `${x(spent)}%`, width: `${Math.min(x(s.need), W - x(spent))}%` }}
              />
            ) : null,
          )}
      </div>
      <div className="num mt-1.5 flex justify-between text-[0.72rem] text-ink-faint">
        <span>0</span>
        <span>{adaptive && budget ? `budget ${fmt(budget, 1)}` : `${fmt(total)} units`}</span>
      </div>
      {segs.length > 1 && (
        <ul className="mt-3 grid gap-1 text-[0.8rem]">
          {segs.map((s, i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="h-2 w-2 flex-none rounded-[2px]" style={{ background: KIND_COLOR[methodKind(s.method)] }} aria-hidden />
              <span className="min-w-0 flex-1 truncate text-ink-soft">{methodLabel(s.method)}</span>
              <span className="num text-ink">{fmt(s.cost)}</span>
            </li>
          ))}
          {adaptive && budget ? (
            <li className="flex items-center gap-2 border-t hairline pt-1">
              <span className="h-2 w-2 flex-none rounded-[2px] bg-sunken" aria-hidden />
              <span className="flex-1 text-ink-faint">Left unspent</span>
              <span className="num text-ink-faint">{fmt(Math.max(budget - total, 0))}</span>
            </li>
          ) : null}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------------------ */
/* Decision path: a vertical stepper of everything the controller did, in order.                                  */
/* ------------------------------------------------------------------------------------------------------------ */

type StepTone = "plain" | "pass" | "fail" | "ghost" | "end" | "lexical" | "semantic" | "expand";

function Step({ icon, title, children, tone = "plain", last }: { icon: ReactNode; title: ReactNode; children?: ReactNode; tone?: StepTone; last?: boolean }) {
  return (
    <li className="relative grid grid-cols-[1.75rem_1fr] gap-x-3 pb-4 last:pb-0">
      {!last && <span className="absolute bottom-0 left-[0.84rem] top-7 w-px bg-line/[0.14]" aria-hidden />}
      <span
        className={clsx(
          "relative flex h-7 w-7 items-center justify-center rounded-full",
          tone === "plain" && "bg-sunken text-ink-soft",
          tone === "lexical" && "bg-accent-soft text-lexical",
          tone === "semantic" && "bg-semantic-soft text-semantic",
          tone === "expand" && "bg-expand-soft text-expand",
          tone === "pass" && "bg-pass-soft text-pass",
          tone === "fail" && "bg-fail-soft text-fail",
          tone === "ghost" && "border border-dashed border-ink-faint/60 text-ink-faint",
          tone === "end" && "bg-ink text-canvas",
        )}
        aria-hidden
      >
        {icon}
      </span>
      <div className="min-w-0 pt-[3px]">
        <div className={clsx("text-[0.88rem] font-medium", tone === "ghost" ? "text-ink-faint" : "text-ink")}>{title}</div>
        {children && <div className="mt-0.5 text-[0.8rem] leading-snug text-ink-soft">{children}</div>}
      </div>
    </li>
  );
}

function routeFacts(f: TraceStep["signals"]): string {
  const out: string[] = [];
  if (typeof f.n_terms === "number") out.push(`${f.n_terms} terms`);
  if (typeof f.mean_idf === "number") out.push(`mean idf ${fmt(f.mean_idf)}`);
  if (f.question) out.push("question form");
  if (f.quotes) out.push("quoted phrase");
  if (f.boolean) out.push("Boolean operators");
  if (typeof f.unknown_share === "number" && f.unknown_share > 0) out.push(`${Math.round(f.unknown_share * 100)}% unknown terms`);
  return out.join(", ");
}

export function DecisionPath({ trace, tau, mode }: { trace: TraceStep[]; tau: number | null; mode: string }) {
  const nodes = buildPath(trace, tau, mode);
  const ico = "h-3.5 w-3.5";
  return (
    <ol className="mt-1" aria-label="Decision path">
      {nodes.map((n, i) => {
        const last = i === nodes.length - 1;
        if (n.kind === "route") {
          const facts = routeFacts(n.features ?? {});
          return (
            <Step key={i} last={last} icon={<Compass className={ico} />} title={<>Route · {n.cls || "query"}</>}>
              {n.why ? `${n.why}. ` : ""}Start with <b className="font-medium text-ink">{methodLabel(n.method)}</b>.
              {facts && <span className="mt-0.5 block text-ink-faint">{facts}</span>}
            </Step>
          );
        }
        if (n.kind === "retrieve") {
          const k = methodKind(n.method);
          return (
            <Step
              key={i}
              last={last}
              tone={k === "semantic" ? "semantic" : k === "expand" ? "expand" : "lexical"}
              icon={k === "semantic" ? <Sparkles className={ico} /> : k === "expand" ? <Wand2 className={ico} /> : <Search className={ico} />}
              title={methodLabel(n.method)}
            >
              <span className="num">
                cost {fmt(n.cost)}, {ms(n.latency)}
                {n.note && !/^\d+ results$/.test(n.note) ? `, ${n.note}` : ""}
              </span>
            </Step>
          );
        }
        if (n.kind === "confidence")
          return (
            <Step
              key={i}
              last={last}
              tone={n.passed === null ? "plain" : n.passed ? "pass" : "fail"}
              icon={<Gauge className={ico} />}
              title={
                <span className="num">
                  Confidence {fmt(n.value)}
                  {n.tau != null && (
                    <span className={n.passed ? "text-pass" : "text-fail"}>
                      {" "}
                      {n.passed ? "≥" : "<"} τ {fmt(n.tau)}
                    </span>
                  )}
                </span>
              }
            >
              {n.passed === null ? "Measured for reference, not used to stop." : n.passed ? "Good enough, stop here." : "Not sure yet, escalate."}
            </Step>
          );
        if (n.kind === "skip")
          return (
            <Step key={i} last={last} tone="ghost" icon={<Ban className={ico} />} title={<>{methodLabel(n.method)} skipped</>}>
              {n.note.replace(/^skip [^:]+: /, "")}
            </Step>
          );
        return (
          <Step
            key={i}
            last={last}
            tone="end"
            icon={n.reason.startsWith("confident") ? <CircleCheck className={ico} /> : n.reason ? <Square className="h-3 w-3" /> : <CircleDot className={ico} />}
            title="Stop"
          >
            {n.reason}
          </Step>
        );
      })}
    </ol>
  );
}
