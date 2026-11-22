import clsx from "clsx";
import { CircleCheck, CircleX } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { isAdaptive, QualityCostChart } from "../components/QualityCostChart";
import { Callout, Segmented, Skeleton, Stat, TableWrap, td, tdNum, th, thNum } from "../components/ui";
import { api, type Collection, type EvaluationResponse, type EvaluationResult, type SystemRow } from "../lib/api";
import { fmt, ms, pct, relChange, signed } from "../lib/format";

const METRICS = ["NDCG@10", "P@5", "Recall@10", "MAP"] as const;
const nd = (s?: SystemRow) => s?.metrics?.["NDCG@10"] ?? null;

type View = "systems" | "types" | "significance" | "ablations" | "risk";

/** One detail table at a time, with a one-line explanation above it. */
function Detail({ title, lead, children, note }: { title: string; lead: ReactNode; children: ReactNode; note?: ReactNode }) {
  return (
    <div data-view={title} className="mt-4">
      <p className="max-w-[80ch] text-[0.86rem] leading-relaxed text-ink-soft">{lead}</p>
      {children}
      {note && <p className="mt-2 text-[0.8rem] text-ink-faint">{note}</p>}
    </div>
  );
}

function SystemsTable({ rows, labelled, highlight }: { rows: SystemRow[]; labelled: boolean; highlight: string }) {
  const bestOf = Object.fromEntries(METRICS.map((m) => [m, Math.max(...rows.map((r) => r.metrics?.[m] ?? -1))]));
  return (
    <TableWrap label="All systems" className="mt-3">
      <thead>
        <tr>
          <th className={th}>System</th>
          {labelled && METRICS.map((m) => <th key={m} className={thNum}>{m}</th>)}
          <th className={thNum}>Mean cost</th>
          <th className={thNum}>Escalated</th>
          <th className={thNum}>Latency p50 / p95</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const on = r.id === highlight;
          return (
            <tr key={r.id} data-highlight={on || undefined} className={clsx(on ? "bg-accent-soft/70" : "hover:bg-raised")}>
              <td className={clsx(td, "min-w-[15rem]")}>
                <span className="num mr-2.5 inline-block w-7 text-[0.8rem] text-ink-faint">{r.id}</span>
                <span className={clsx(on && "font-semibold", !isAdaptive(r.id) && !on && "text-ink-soft")}>{r.label}</span>
              </td>
              {labelled &&
                METRICS.map((m) => {
                  const v = r.metrics?.[m];
                  return (
                    <td key={m} className={clsx(tdNum, v != null && v === bestOf[m] && "font-semibold")}>
                      {fmt(v, 3)}
                    </td>
                  );
                })}
              <td className={tdNum}>{fmt(r.mean_cost)}</td>
              <td className={tdNum}>{isAdaptive(r.id) ? pct(r.escalation_rate) : "—"}</td>
              <td className={clsx(tdNum, "whitespace-nowrap text-ink-soft")}>
                {ms(r.latency_median_ms)} / {ms(r.latency_p95_ms)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </TableWrap>
  );
}

/** NDCG@10 and cost per query type for the key systems: where adaptivity pays off. */
function byTypeData(r: EvaluationResult) {
  const sys = ["S2", "S4", "S5", "S7"].map((id) => r.systems.find((s) => s.id === id)).filter((s): s is SystemRow => !!s);
  const types = Array.from(new Set(sys.flatMap((s) => Object.keys(s.by_type ?? {}))));
  return { sys, types };
}

function ByTypeTable({ r }: { r: EvaluationResult }) {
  const { sys, types } = byTypeData(r);
  return (
    <TableWrap label="By query type" className="mt-3">
      <thead>
        <tr>
          <th className={th}>System</th>
          {types.map((t) => (
            <th key={t} className={thNum}>
              <span className="capitalize">{t}</span>
              <span className="ml-1 font-normal normal-case text-ink-faint">({sys[0].by_type?.[t]?.n ?? "?"} queries)</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sys.map((s) => (
          <tr key={s.id} data-highlight={s.id === "S7" || undefined} className={s.id === "S7" ? "bg-accent-soft/70" : "hover:bg-raised"}>
            <td className={clsx(td, "min-w-[11rem]", s.id === "S7" && "font-semibold")}>{s.label}</td>
            {types.map((t) => {
              const v = s.by_type?.[t]?.["NDCG@10"] ?? null;
              const best = Math.max(...sys.map((x) => x.by_type?.[t]?.["NDCG@10"] ?? -1));
              return (
                <td key={t} className={clsx(tdNum, "whitespace-nowrap")}>
                  <span className={clsx(v === best && "font-semibold")}>{fmt(v, 3)}</span>
                  <span className="text-ink-faint"> · {fmt(s.by_type?.[t]?.mean_cost, 1)}u</span>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </TableWrap>
  );
}

export function EvaluationPage({ collection }: { collection: Collection }) {
  const [split, setSplit] = useState<string | undefined>(undefined);
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("systems");

  useEffect(() => {
    setData(null);
    setSplit(undefined);
  }, [collection.id]);
  useEffect(() => {
    let live = true;
    setError(null);
    api.evaluation(collection.id, split).then(
      (d) => live && setData(d),
      (e) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [collection.id, split]);

  if (error) return <Callout tone="warn" title="Could not load results.">{error}</Callout>;
  if (!data) return <Skeleton className="mt-2 h-72" />;
  const r = data.result;
  if (!r)
    return (
      <Callout tone="plain" title="No saved results yet.">
        Run <code>python experiments/fit_and_tune.py {collection.id}</code> then{" "}
        <code>python experiments/run_all.py {collection.id} --split test</code>.
      </Callout>
    );

  const sys = Object.fromEntries(r.systems.map((s) => [s.id, s]));
  const ctl = sys.S7;
  const hyb = sys.S5;
  const bm25 = sys.S2;
  const s8 = sys.S8;
  const certified = r.risk_curve.filter((x) => x.certified_on_val);
  const held = certified.filter((x) => x.test_failure_rate != null && x.test_failure_rate <= x.delta);
  const vsHybrid = r.significance.filter((s) => s.b === "S5" && isAdaptive(s.a));
  const keep = ctl && hyb && nd(hyb) ? (nd(ctl) ?? 0) / (nd(hyb) as number) : null;
  const saving = ctl && hyb ? relChange(hyb.mean_cost ?? 0, ctl.mean_cost ?? 0) : null;

  const views: { value: View; label: string }[] = [{ value: "systems", label: "All systems" }];
  if (r.labelled && byTypeData(r).types.length >= 2) views.push({ value: "types", label: "By query type" });
  if (r.labelled && vsHybrid.length > 0) views.push({ value: "significance", label: "Significance" });
  if (r.ablations.length > 0) views.push({ value: "ablations", label: "Ablations" });
  if (r.labelled && r.risk_curve.length > 0) views.push({ value: "risk", label: "Risk control" });
  const shown = views.some((v) => v.value === view) ? view : "systems";

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-[0.88rem] text-ink-soft">
          Saved results for <b className="num font-medium text-ink">{r.n_queries}</b> {r.split} queries, run{" "}
          {r.timestamp_utc.replace(/T(\d\d)(\d\d).*/, " at $1:$2 UTC")}
          {r.hardware.cpu_count ? ` on ${r.hardware.cpu_count} CPUs` : ""}.
        </p>
        {data.available.length > 1 && (
          <Segmented
            label="Query split"
            size="sm"
            value={r.split}
            onChange={setSplit}
            options={data.available.map((s) => ({ value: s, label: s === "val" ? "Validation" : s === "test" ? "Test" : s }))}
          />
        )}
      </div>

      {!r.labelled ? (
        <Callout tone="warn" title="No relevance labels yet.">
          This collection's benchmark has not been judged, so only cost and latency are shown. Grade the pool on the Judge page, then run{" "}
          <code>python experiments/run_all.py {collection.id} --split test</code>.
        </Callout>
      ) : (
        <>
          {keep != null && saving != null && (
            <p className="mt-6 max-w-[34ch] text-[clamp(1.35rem,2.6vw,1.9rem)] font-semibold leading-[1.2] tracking-[-0.02em] text-ink">
              The controller keeps {pct(keep)} of the fixed hybrid's quality and spends {pct(Math.abs(saving))}{" "}
              {saving <= 0 ? "less" : "more"} per query.
            </p>
          )}
          <div className="panel mt-6 grid grid-cols-2 divide-line/[0.08] lg:grid-cols-4 lg:divide-x [&>*]:px-5 [&>*]:py-4">
            <Stat
              label="Controller quality"
              value={fmt(nd(ctl), 3)}
              sub="NDCG@10"
              note={hyb && ctl ? `${pct((nd(ctl) ?? 0) / (nd(hyb) || 1))} of the fixed hybrid's ${fmt(nd(hyb), 3)}` : undefined}
            />
            <Stat
              label="Cost vs fixed hybrid"
              value={ctl && hyb ? pct(relChange(hyb.mean_cost ?? 0, ctl.mean_cost ?? 0)) : "—"}
              note={ctl && hyb ? `${fmt(ctl.mean_cost, 1)} vs ${fmt(hyb.mean_cost, 1)} units per query` : undefined}
              tone={saving != null && saving < 0 ? "pass" : undefined}
            />
            <Stat
              label="vs BM25 alone"
              value={ctl && bm25 ? signed(nd(ctl)! - nd(bm25)!) : "—"}
              sub="NDCG@10"
              note={bm25 ? `BM25 from scratch scores ${fmt(nd(bm25), 3)} at cost 1` : undefined}
            />
            <Stat
              label="Risk control held"
              value={held.length}
              sub={`of ${certified.length}`}
              note="certified risk levels where the test failure rate stayed at or below δ"
              tone={certified.length && held.length === certified.length ? "pass" : undefined}
            />
          </div>
          <QualityCostChart result={r} />
        </>
      )}

      <div className="mt-12 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-[1.15rem] font-semibold tracking-[-0.01em]">Details</h2>
        <Segmented label="Detail view" size="sm" value={shown} onChange={setView} options={views} />
      </div>

      {shown === "systems" && (
        <Detail
          title="All systems"
          lead={
            <>
              S0–S5 are fixed methods, S6–S8 the adaptive controller and N1–N4 the ContextWeave+ extensions. Cost is in BM25 queries. The
              highlighted row is the full controller; bold is the best in its column.
            </>
          }
        >
          <SystemsTable rows={r.systems} labelled={r.labelled} highlight="S7" />
        </Detail>
      )}

      {shown === "types" && (
        <Detail
          title="By query type"
          lead="The router sorts queries into classes. Fixed methods win some classes and lose others; the controller should track the best in each while spending only what that class needs."
          note="Each cell: NDCG@10 · mean cost in BM25 units. Bold is the best in its column."
        >
          <ByTypeTable r={r} />
        </Detail>
      )}

      {shown === "significance" && (
        <Detail
          title="Is the difference real?"
          lead="Paired t-test against the fixed hybrid (S5) on the same queries, Holm-corrected. An interval that crosses zero means no significant loss."
        >
          <TableWrap label="Significance tests" className="mt-3">
            <thead>
              <tr>
                <th className={th}>System</th>
                <th className={thNum}>Δ NDCG@10 vs S5</th>
                <th className={thNum}>95% CI</th>
                <th className={thNum}>p (Holm)</th>
                <th className={th}>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {vsHybrid.map((s) => {
                const sig = s.p_holm < 0.05;
                return (
                  <tr key={s.a} className="hover:bg-raised">
                    <td className={clsx(td, "min-w-[14rem]")}>
                      <span className="num mr-2.5 inline-block w-7 text-[0.8rem] text-ink-faint">{s.a}</span>
                      {sys[s.a]?.label}
                    </td>
                    <td className={tdNum}>{signed(s.mean_diff)}</td>
                    <td className={clsx(tdNum, "whitespace-nowrap text-ink-soft")}>
                      [{fmt(s.ci95[0], 3)}, {fmt(s.ci95[1], 3)}]
                    </td>
                    <td className={tdNum}>{s.p_holm < 0.001 ? "< 0.001" : fmt(s.p_holm, 3)}</td>
                    <td className={clsx(td, "text-[0.84rem] font-medium", sig ? "text-fail" : "text-pass")}>
                      {sig ? "significant" : "no significant loss"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </TableWrap>
        </Detail>
      )}

      {shown === "ablations" && (
        <Detail title="What each part contributes" lead="Ablations: the full controller (S8) with one component switched off at a time.">
          <TableWrap label="Ablations" className="mt-3">
            <thead>
              <tr>
                <th className={th}>Variant</th>
                {r.labelled && <th className={thNum}>NDCG@10</th>}
                {r.labelled && <th className={thNum}>Δ vs S8</th>}
                <th className={thNum}>Mean cost</th>
                <th className={thNum}>Δ cost vs S8</th>
              </tr>
            </thead>
            <tbody>
              {r.ablations.map((a) => {
                const d = nd(a) != null && nd(s8) != null ? nd(a)! - nd(s8)! : null;
                return (
                  <tr key={a.id} className="hover:bg-raised">
                    <td className={clsx(td, "min-w-[13rem]")}>{a.label.replace("S8 - ", "without ")}</td>
                    {r.labelled && <td className={tdNum}>{fmt(nd(a), 3)}</td>}
                    {r.labelled && (
                      <td className={clsx(tdNum, d != null && d < -0.005 && "text-fail")}>{d != null ? signed(d) : "—"}</td>
                    )}
                    <td className={tdNum}>{fmt(a.mean_cost)}</td>
                    <td className={tdNum}>{s8?.mean_cost ? pct(relChange(s8.mean_cost, a.mean_cost ?? 0)) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </TableWrap>
        </Detail>
      )}

      {shown === "risk" && (
        <Detail
          title="Risk-controlled stopping"
          lead="Learn-then-Test picks τ on validation queries so that, with 90% confidence, at most δ of queries lose more than 0.05 NDCG@10 against the fixed hybrid. The test split checks whether that promise held."
        >
          <TableWrap label="Risk control" className="mt-3">
            <thead>
              <tr>
                <th className={thNum}>Allowed failure rate δ</th>
                <th className={thNum}>τ chosen</th>
                <th className={thNum}>Test failure rate</th>
                <th className={thNum}>Test NDCG@10</th>
                <th className={thNum}>Test cost</th>
                <th className={th}>Guarantee</th>
              </tr>
            </thead>
            <tbody>
              {r.risk_curve.map((x) => {
                const ok = x.certified_on_val && x.test_failure_rate != null && x.test_failure_rate <= x.delta;
                return (
                  <tr key={x.delta} className="hover:bg-raised">
                    <td className={tdNum}>{pct(x.delta)}</td>
                    <td className={tdNum}>{x.certified_on_val ? fmt(x.tau) : "—"}</td>
                    <td className={tdNum}>{pct(x.test_failure_rate, 1)}</td>
                    <td className={tdNum}>{fmt(x.test_ndcg, 3)}</td>
                    <td className={tdNum}>{fmt(x.test_cost, 1)}</td>
                    <td className={td}>
                      {!x.certified_on_val ? (
                        <span className="text-[0.82rem] text-ink-faint">not certifiable</span>
                      ) : ok ? (
                        <span className="inline-flex items-center gap-1.5 text-[0.84rem] font-medium text-pass">
                          <CircleCheck className="h-4 w-4" /> held
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-[0.84rem] font-medium text-fail">
                          <CircleX className="h-4 w-4" /> violated
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </TableWrap>
        </Detail>
      )}
    </div>
  );
}
