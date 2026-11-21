import clsx from "clsx";
import { useLayoutEffect, useRef, useState } from "react";
import type { EvaluationResult, SystemRow } from "../lib/api";
import { fmt } from "../lib/format";

// Colours come from the theme (index.css), so the chart follows light and dark mode: adaptive systems in the
// accent, fixed baselines in the de-emphasis gray, the tau/budget sweep as a faint accent wash.
const LABELLED = new Set(["S2", "S4", "S5", "S7"]);
export const isAdaptive = (id: string) => /^(S6|S7|S8|N\d)$/.test(id);

const LOG_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(720);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setW(Math.max(280, el.clientWidth));
    update();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

type Pt = { x: number; y: number; cost: number; ndcg: number; label: string; id?: string; adaptive?: boolean; sweep?: boolean };

type Box = { x0: number; x1: number; y0: number; y1: number };
export type PlacedLabel = {
  id: string; text: string; tx: number; ty: number; anchor: "start" | "end" | "middle";
  px: number; py: number; lx: number; ly: number; leader: boolean; box: Box;
};

const overlap = (a: Box, b: Box) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));

/** Direct labels for the key systems, each at the candidate spot (right, left, above, below - near first, then
 *  further out with a leader line) that overlaps the fewest other points and earlier labels and stays in the plot. */
export function placeLabels(pts: Pt[], plot: Box): PlacedLabel[] {
  const out: PlacedLabel[] = [];
  const dots = pts.map((p) => ({ x0: p.x - 7, x1: p.x + 7, y0: p.y - 7, y1: p.y + 7 }));
  for (const p of pts.filter((q) => LABELLED.has(q.id!))) {
    const text = `${p.id} ${p.label.replace("(from scratch)", "").trim()}`;
    const w = text.length * 6.4;
    const h = 14;
    const cands: { dx: number; dy: number; anchor: PlacedLabel["anchor"] }[] = [];
    for (const d of [10, 28]) {
      cands.push({ dx: d, dy: 0, anchor: "start" }, { dx: -d, dy: 0, anchor: "end" });
      cands.push({ dx: 0, dy: -d - 2, anchor: "middle" }, { dx: 0, dy: d + 2, anchor: "middle" });
      cands.push({ dx: d, dy: -d, anchor: "start" }, { dx: d, dy: d, anchor: "start" }, { dx: -d, dy: -d, anchor: "end" }, { dx: -d, dy: d, anchor: "end" });
    }
    let best: PlacedLabel | null = null;
    let bestCost = Infinity;
    cands.forEach((c, i) => {
      const tx = p.x + c.dx;
      const ty = p.y + c.dy;
      const bx0 = c.anchor === "start" ? tx : c.anchor === "end" ? tx - w : tx - w / 2;
      const box = { x0: bx0, x1: bx0 + w, y0: ty - h / 2, y1: ty + h / 2 };
      const outside = box.x0 < plot.x0 || box.x1 > plot.x1 || box.y0 < plot.y0 || box.y1 > plot.y1 ? 1e6 : 0;
      const cost = outside + dots.reduce((s, d) => s + overlap(box, d), 0) * 3 + out.reduce((s, l) => s + overlap(box, l.box), 0) * 5 + i * 0.5;
      if (cost < bestCost) {
        bestCost = cost;
        const far = Math.abs(c.dx) > 12 || Math.abs(c.dy) > 14;
        best = { id: p.id!, text, tx, ty, anchor: c.anchor, px: p.x, py: p.y, lx: c.anchor === "start" ? tx - 2 : c.anchor === "end" ? tx + 2 : tx, ly: c.anchor === "middle" ? ty + (c.dy < 0 ? h / 2 : -h / 2) : ty, leader: far, box };
      }
    });
    if (best) out.push(best);
  }
  return out;
}

/** Split systems into the plotted ones and those far below the rest (e.g. Boolean AND), which would flatten the chart. */
export function plotted(systems: SystemRow[]) {
  const withQ = systems.filter((s) => s.metrics?.["NDCG@10"] != null && s.mean_cost != null && s.mean_cost > 0);
  const vals = withQ.map((s) => s.metrics!["NDCG@10"] as number).sort((a, b) => a - b);
  const median = vals[Math.floor(vals.length / 2)] ?? 0;
  const keep = withQ.filter((s) => (s.metrics!["NDCG@10"] as number) >= 0.6 * median);
  const off = withQ.filter((s) => !keep.includes(s));
  return { keep, off };
}

export function QualityCostChart({ result }: { result: EvaluationResult }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<Pt | null>(null);
  const { keep, off } = plotted(result.systems);
  const sweep = result.sweep.filter((p) => p.ndcg != null && p.cost > 0);

  const H = 340;
  const m = { top: 16, right: 24, bottom: 46, left: 52 };
  const iw = width - m.left - m.right;
  const ih = H - m.top - m.bottom;

  const costs = [...keep.map((s) => s.mean_cost as number), ...sweep.map((p) => p.cost)];
  const ys = [...keep.map((s) => s.metrics!["NDCG@10"] as number), ...sweep.map((p) => p.ndcg as number)];
  const x0 = Math.min(...costs) * 0.75;
  const x1 = Math.max(...costs) * 1.35;
  const y0 = Math.floor((Math.min(...ys) - 0.01) * 50) / 50;
  const y1 = Math.ceil((Math.max(...ys) + 0.01) * 50) / 50;
  const sx = (c: number) => m.left + ((Math.log10(c) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * iw;
  const sy = (v: number) => m.top + (1 - (v - y0) / (y1 - y0)) * ih;
  const xticks = LOG_TICKS.filter((t) => t >= x0 && t <= x1);
  const step = y1 - y0 > 0.2 ? 0.05 : 0.02;
  const yticks: number[] = [];
  for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) yticks.push(Number(v.toFixed(3)));

  const pts: Pt[] = keep.map((s) => {
    const cost = s.mean_cost as number;
    const ndcg = s.metrics!["NDCG@10"] as number;
    return { x: sx(cost), y: sy(ndcg), cost, ndcg, label: s.label, id: s.id, adaptive: isAdaptive(s.id) };
  });
  const sweepPts: Pt[] = sweep.map((p) => ({
    x: sx(p.cost),
    y: sy(p.ndcg as number),
    cost: p.cost,
    ndcg: p.ndcg as number,
    label: `S8 with τ ${fmt(p.tau)}, budget ${fmt(p.budget, 1)}`,
    sweep: true,
  }));
  const labels = placeLabels([...pts, ...sweepPts], { x0: m.left, x1: width - m.right, y0: m.top, y1: H - m.bottom });

  return (
    <figure className="panel mt-4 p-5" aria-label="Quality versus cost">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-3">
        <span>
          <span className="font-semibold">Quality against cost per query</span>
          <span className="block text-[0.82rem] text-ink-soft">Up and to the left is better: more NDCG@10 for less effort.</span>
        </span>
        <span className="flex flex-wrap gap-4 text-[0.8rem] text-ink-soft" aria-label="Legend">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-accent" />
            Adaptive controller
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-ink-faint" />
            Fixed method
          </span>
          {sweepPts.length > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-accent/40" />
              τ and budget sweep
            </span>
          )}
        </span>
      </figcaption>
      <div ref={ref} className="relative mt-4">
        <svg width={width} height={H} role="img" aria-label="Scatter plot of NDCG@10 against mean cost per query" onMouseLeave={() => setHover(null)} className="overflow-visible">
          {yticks.map((t) => (
            <g key={`y${t}`}>
              <line x1={m.left} x2={width - m.right} y1={sy(t)} y2={sy(t)} className="stroke-line/[0.08]" strokeWidth={1} />
              <text x={m.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" fontSize="11" className="num fill-ink-faint">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {xticks.map((t) => (
            <g key={`x${t}`}>
              <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={H - m.bottom} className="stroke-line/[0.08]" strokeWidth={1} />
              <text x={sx(t)} y={H - m.bottom + 16} textAnchor="middle" fontSize="11" className="num fill-ink-faint">
                {t}
              </text>
            </g>
          ))}
          <text x={m.left + iw / 2} y={H - 8} textAnchor="middle" fontSize="11.5" className="fill-ink-soft">
            Mean cost per query, in BM25 units (log scale)
          </text>
          <text transform={`translate(13 ${m.top + ih / 2}) rotate(-90)`} textAnchor="middle" fontSize="11.5" className="fill-ink-soft">
            NDCG@10
          </text>

          {sweepPts.map((p, i) => (
            <circle key={`s${i}`} cx={p.x} cy={p.y} r={3} className="fill-accent" opacity={0.22} onMouseEnter={() => setHover(p)} />
          ))}
          {pts.map((p) => (
            <g key={p.id} onMouseEnter={() => setHover(p)} onFocus={() => setHover(p)} onBlur={() => setHover(null)} tabIndex={0} role="button" aria-label={`${p.id} ${p.label}: NDCG@10 ${fmt(p.ndcg, 3)}, cost ${fmt(p.cost)}`} className="outline-none">
              <circle cx={p.x} cy={p.y} r={14} fill="transparent" />
              <circle cx={p.x} cy={p.y} r={hover === p ? 7 : 5.5} className={clsx(p.adaptive ? "fill-accent" : "fill-ink-faint", "stroke-surface transition-[r]")} strokeWidth={2} />
            </g>
          ))}
          {labels.map((l) => (
            <g key={`l${l.id}`} pointerEvents="none">
              {l.leader && <line x1={l.px} y1={l.py} x2={l.lx} y2={l.ly} className="stroke-ink-faint" strokeWidth={1} />}
              <text x={l.tx} y={l.ty} dy="0.32em" textAnchor={l.anchor} fontSize="11.5" fontWeight={600} className="fill-ink stroke-surface" paintOrder="stroke" strokeWidth={3}>
                {l.text}
              </text>
            </g>
          ))}
        </svg>
        {hover && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-[16rem] rounded-ctl border hairline bg-surface px-3 py-2 text-[0.8rem] shadow-lift"
            style={{ left: Math.min(hover.x + 12, width - 200), top: Math.max(hover.y - 56, 0) }}
            role="status"
          >
            <div className="font-medium">{hover.id ? `${hover.id} ${hover.label}` : hover.label}</div>
            <div className="num text-ink-soft">
              NDCG@10 {fmt(hover.ndcg, 3)}, cost {fmt(hover.cost)}
            </div>
          </div>
        )}
      </div>
      {off.length > 0 && (
        <p className="mt-2 text-[0.8rem] text-ink-faint">
          Off the scale and listed in the table: {off.map((s) => `${s.id} ${s.label} (NDCG@10 ${fmt(s.metrics!["NDCG@10"], 3)})`).join(", ")}.
        </p>
      )}
    </figure>
  );
}
