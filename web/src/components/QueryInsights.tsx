import clsx from "clsx";
import type { SearchResponse } from "../lib/api";
import { compact, fmt, methodKind, methodLabel, ms, pct } from "../lib/format";
import { Badge, Callout, Meter, MethodTag, Panel, TableWrap, td, tdNum, th, thNum } from "./ui";

const SIGNAL_HELP: Record<string, string> = {
  margin: "how far the top results stand out from the 10th",
  coverage: "share of the IDF-weighted query terms the top 5 contain",
  concentration: "how peaked the score distribution is",
  agreement: "overlap of the lexical and semantic top lists",
  nqc: "normalised query commitment (QPP)",
  wig: "weighted information gain (QPP)",
};

/** The query as the index sees it: stemmed terms with document frequency and IDF, unknown terms flagged. */
export function TermChips({ resp }: { resp: SearchResponse }) {
  const terms = resp.query.terms;
  const maxIdf = Math.max(...terms.map((t) => t.idf ?? 0), 1e-9);
  return (
    <div aria-label="How the query was read">
      {terms.length === 0 && <p className="text-[0.85rem] text-ink-soft">Only stop words, so nothing is left to match on.</p>}
      <ul className="grid gap-1.5">
        {terms.map((t) => (
          <li key={t.term} className="grid grid-cols-[minmax(0,7.5rem)_1fr_auto] items-center gap-3 text-[0.82rem]">
            <code className={clsx("truncate text-[0.78rem]", !t.known && "bg-fail-soft text-fail")}>{t.term}</code>
            {t.known ? <Meter value={t.idf / maxIdf} tone="lexical" label={`${t.term} idf`} /> : <span className="text-[0.78rem] text-fail">not in the collection</span>}
            <span className="num whitespace-nowrap text-right text-[0.76rem] text-ink-faint">{t.known ? `df ${compact(t.df)}, idf ${fmt(t.idf)}` : ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function QueryInsights({ resp }: { resp: SearchResponse }) {
  const ran = resp.methods_run.filter((m) => resp.lists[m]);
  const conf = resp.trace.filter((s) => s.action === "confidence" && s.confidence != null);
  const signalKeys = Array.from(
    new Set(conf.flatMap((s) => Object.keys(s.signals).filter((k) => !k.startsWith("alpha") && typeof s.signals[k] === "number"))),
  );
  const finalRank = new Map(resp.results.map((r) => [r.passage_id, r.rank]));
  const titleOf = new Map<string, string>();
  for (const l of Object.values(resp.lists)) for (const x of l.top) titleOf.set(x.passage_id, x.title);
  for (const r of resp.results) titleOf.set(r.passage_id, r.title);

  return (
    <div className="border-t hairline">
      <Panel
        title="What each method retrieved"
        aside={
          resp.agreement ? (
            <span className="num text-[0.84rem] text-ink-soft">
              Lexical and semantic top 10 share <b className="font-medium text-ink">{resp.agreement.overlap}</b> of {resp.agreement.of}
            </span>
          ) : undefined
        }
        lead={
          ran.length > 1
            ? "Every method that ran produced its own ranked list, and the controller fused them. Lexical methods match words and dense retrieval matches meaning, so they often disagree. The right-hand column shows where each passage ended up in the final results."
            : "Only one method ran for this query, so its list is the result list."
        }
      >
        <div className={clsx("mt-4 grid gap-4", ran.length > 1 && "lg:grid-cols-2")}>
          {ran.map((m) => {
            const l = resp.lists[m];
            const st = l.stats;
            const kept = l.top.slice(0, 10).filter((x) => finalRank.has(x.passage_id)).length;
            return (
              <div key={m} className="panel min-w-0 overflow-hidden">
                <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b hairline px-4 py-3">
                  <MethodTag method={m} className="font-medium" />
                  <span className="num text-[0.78rem] text-ink-faint">
                    {[
                      st.passages_scored != null && st.N ? `${pct(st.passages_scored / st.N)} of passages scored` : null,
                      st.postings_touched != null ? `${compact(st.postings_touched)} postings` : null,
                      ms(st.latency_ms),
                    ]
                      .filter(Boolean)
                      .join(", ")}
                  </span>
                </div>
                {l.eliminated_terms.length > 0 && (
                  <p className="border-b hairline bg-raised px-4 py-2 text-[0.78rem] text-ink-soft">
                    Index elimination dropped low-IDF terms: {l.eliminated_terms.join(", ")}
                  </p>
                )}
                <ol className="divide-y divide-line/[0.07]">
                  {l.top.slice(0, 8).map((x) => {
                    const fr = finalRank.get(x.passage_id);
                    return (
                      <li key={x.passage_id} className="grid grid-cols-[1.5rem_1fr_auto_3.25rem] items-center gap-3 px-4 py-2 text-[0.84rem]">
                        <span className="num text-ink-faint">{x.rank}</span>
                        <span className="truncate" title={x.title}>
                          {x.title}
                        </span>
                        <span className="num text-[0.78rem] text-ink-soft">{fmt(x.score, 3)}</span>
                        <span className="text-right">
                          {fr ? (
                            <Badge tone={methodKind(m) === "semantic" ? "semantic" : "accent"} className="num">
                              #{fr}
                            </Badge>
                          ) : (
                            <span className="text-[0.78rem] text-ink-faint" title="not in the final results">
                              out
                            </span>
                          )}
                        </span>
                      </li>
                    );
                  })}
                  {l.top.length === 0 && <li className="px-4 py-3 text-[0.84rem] text-ink-faint">No passages matched.</li>}
                </ol>
                {ran.length > 1 && l.top.length > 0 && (
                  <p className="num border-t hairline px-4 py-2 text-[0.78rem] text-ink-faint">
                    {kept} of its top {Math.min(10, l.top.length)} made the final results
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </Panel>

      {resp.expansion && (
        <Panel
          title="Query expansion"
          aside={
            <Badge tone={resp.expansion.added_terms.length ? "expand" : "plain"}>
              {resp.expansion.added_terms.length ? `${resp.expansion.added_terms.length} terms added` : "no change"}
            </Badge>
          }
          lead="Pseudo-relevance feedback. Candidate terms come from the top passages, and an embedding filter keeps only those close in meaning to the query so the query does not drift."
        >
          {resp.expansion.added_terms.length ? (
            <div className="mt-4 grid gap-4 md:grid-cols-[1fr_1fr]">
              <div className="panel min-w-0 p-4">
                <div className="label mb-2">Added terms</div>
                <ul className="grid gap-1.5">
                  {resp.expansion.added_terms.map((a) => (
                    <li key={String(a.term)} className="grid grid-cols-[minmax(0,8rem)_1fr_auto] items-center gap-3 text-[0.84rem]">
                      <span className="truncate font-medium text-expand">+ {String(a.word ?? a.term)}</span>
                      <span className="num text-[0.78rem] text-ink-faint">
                        {typeof a.tfidf === "number" ? `weight ${fmt(a.tfidf)}` : ""}
                      </span>
                      <span className="num text-right text-[0.78rem] text-ink-soft">
                        {typeof a.similarity === "number" ? `similarity ${fmt(a.similarity)}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              {resp.expansion.feedback.length > 0 && (
                <div className="panel min-w-0 p-4">
                  <div className="label mb-2">Feedback passages the terms came from</div>
                  <ol className="grid gap-1.5 text-[0.84rem]">
                    {resp.expansion.feedback.map((pid, i) => (
                      <li key={pid} className="grid grid-cols-[1.25rem_1fr] gap-2">
                        <span className="num text-ink-faint">{i + 1}</span>
                        <span className="truncate" title={pid}>
                          {titleOf.get(pid) ?? <span className="font-mono text-[0.78rem]">{pid}</span>}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ) : (
            <Callout tone="plain">No candidate passed the similarity filter, so the query was left unchanged.</Callout>
          )}
        </Panel>
      )}

      {conf.length > 0 && (
        <Panel
          title="Why the controller was or was not confident"
          lead={
            <>
              confidence = sigmoid(bias + Σ wᵢ·signalᵢ), recomputed after every step and compared with τ ={" "}
              <b className="num font-medium text-ink">{fmt(resp.tau)}</b>. The weights live in <code>config/settings.yaml</code>.
            </>
          }
        >
          <TableWrap label="Confidence signals per step">
            <thead>
              <tr>
                <th className={th}>After</th>
                {signalKeys.map((k) => (
                  <th key={k} className={thNum} title={SIGNAL_HELP[k]}>
                    {k}
                  </th>
                ))}
                <th className={thNum}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {conf.map((s, i) => {
                const after = resp.trace
                  .slice(0, resp.trace.indexOf(s))
                  .filter((t) => t.action === "retrieve")
                  .map((t) => methodLabel(t.method));
                const ok = resp.tau != null && (s.confidence ?? 0) >= resp.tau;
                return (
                  <tr key={i}>
                    <td className={clsx(td, "min-w-[10rem]")}>{after.join(" + ") || "—"}</td>
                    {signalKeys.map((k) => {
                      const v = s.signals[k] as number | null | undefined;
                      return (
                        <td key={k} className={clsx(tdNum, "text-ink-soft")}>
                          <span className="inline-flex items-center justify-end gap-2">
                            {v != null && <Meter value={v} tone="mute" className="w-10" />}
                            {fmt(v)}
                          </span>
                        </td>
                      );
                    })}
                    <td className={clsx(tdNum, "font-semibold", ok ? "text-pass" : "text-fail")}>{fmt(s.confidence)}</td>
                  </tr>
                );
              })}
            </tbody>
          </TableWrap>
          <dl className="mt-3 grid gap-x-8 gap-y-1 text-[0.8rem] text-ink-soft sm:grid-cols-2">
            {signalKeys
              .filter((k) => SIGNAL_HELP[k])
              .map((k) => (
                <div key={k}>
                  <dt className="inline font-medium text-ink">{k}</dt>: <dd className="inline">{SIGNAL_HELP[k]}</dd>
                </div>
              ))}
          </dl>
        </Panel>
      )}
    </div>
  );
}
