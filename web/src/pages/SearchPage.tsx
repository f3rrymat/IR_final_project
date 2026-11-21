import clsx from "clsx";
import { ChevronDown, LoaderCircle, RotateCcw, Search as SearchIcon, SlidersHorizontal } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { DecisionPath, SpendLedger } from "../components/DecisionPath";
import { QueryInsights, TermChips } from "../components/QueryInsights";
import { ResultCard } from "../components/ResultCard";
import { Badge, Callout, Chip, kindBg, MethodTag, Skeleton, Stat, Toggle } from "../components/ui";
import { api, type Collection, type FeatureFlag, type Mode, type SearchResponse } from "../lib/api";
import { fmt, methodLabel, ms, parseRoute, stopSentence } from "../lib/format";

const PRIMARY: Mode[] = ["auto", "bm25", "dense", "hybrid"];
const OTHER: Mode[] = ["tfidf", "boolean", "bm25_champion", "route"];
const MODE_HINT: Record<Mode, string> = {
  auto: "Routes the query, checks confidence, escalates only when needed",
  bm25: "Okapi BM25 over the inverted index",
  dense: "MiniLM sentence embeddings + FAISS inner product",
  hybrid: "Always BM25 + dense with a fixed α, the expensive baseline",
  tfidf: "ltc.ltc cosine",
  boolean: "Pure Boolean AND / OR / NOT",
  bm25_champion: "BM25 on champion lists with index elimination",
  route: "The router's first choice only, no escalation",
};
const FEATURES: { id: FeatureFlag; label: string; hint: string }[] = [
  { id: "qpp_signals", label: "QPP confidence signals", hint: "Adds NQC and WIG query-performance predictors" },
  { id: "voc_policy", label: "Value-of-computation escalation", hint: "Takes the step with the best expected gain per cost" },
  { id: "risk_controlled_tau", label: "Risk-controlled τ", hint: "τ certified with Learn-then-Test on validation queries" },
  { id: "qpp_fusion", label: "QPP-percentile fusion", hint: "Sets α(q) from the two lists' predicted quality" },
];

interface Settings {
  budget: number;
  tau: number;
  expansion: boolean;
  features: Record<FeatureFlag, boolean>;
  domains: string[];
}

const defaultsFor = (c: Collection): Settings => ({
  budget: Math.min(c.defaults.budget, c.max_budget),
  tau: c.defaults.tau,
  expansion: c.expansion_default,
  features: { qpp_signals: false, voc_policy: false, risk_controlled_tau: false, qpp_fusion: false },
  domains: [],
});

function readUrl(): { q: string; mode: Mode } {
  const p = new URLSearchParams(window.location.search);
  const m = p.get("mode") as Mode | null;
  return { q: p.get("q") ?? "", mode: m && [...PRIMARY, ...OTHER].includes(m) ? m : "auto" };
}

export function SearchPage({ collection }: { collection: Collection }) {
  const init = readUrl();
  const [input, setInput] = useState(init.q);
  const [mode, setMode] = useState<Mode>(init.mode);
  const [settings, setSettings] = useState<Settings>(() => defaultsFor(collection));
  const [showSettings, setShowSettings] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [resp, setResp] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const lastQuery = useRef(init.q);

  const run = useCallback(
    async (q: string, m: Mode, s: Settings) => {
      const query = q.trim();
      if (!query) return;
      lastQuery.current = query;
      abort.current?.abort();
      const ctl = new AbortController();
      abort.current = ctl;
      setLoading(true);
      setError(null);
      const slowTimer = window.setTimeout(() => setSlow(true), 1500);
      const url = new URL(window.location.href);
      url.searchParams.set("q", query);
      if (m === "auto") url.searchParams.delete("mode");
      else url.searchParams.set("mode", m);
      window.history.replaceState(null, "", url);
      try {
        const r = await api.search(
          {
            collection: collection.id,
            query,
            mode: m,
            budget: s.budget,
            tau: s.tau,
            expansion: s.expansion,
            features: s.features,
            domains: s.domains,
          },
          ctl.signal,
        );
        setResp(r);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError((e as Error).message);
      } finally {
        window.clearTimeout(slowTimer);
        if (abort.current === ctl) {
          setLoading(false);
          setSlow(false);
        }
      }
    },
    [collection.id],
  );

  // new collection: its own tuned defaults, and re-run the current query there
  useEffect(() => {
    const s = defaultsFor(collection);
    setSettings(s);
    setResp(null);
    if (lastQuery.current) run(lastQuery.current, mode, s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collection.id]);

  const submit = (q = input, m = mode, s = settings) => {
    setInput(q);
    run(q, m, s);
  };
  const changeMode = (m: Mode) => {
    setMode(m);
    if (lastQuery.current) run(lastQuery.current, m, settings);
  };
  const changeSettings = (patch: Partial<Settings>) => {
    const s = { ...settings, ...patch };
    setSettings(s);
    if (lastQuery.current) run(lastQuery.current, mode, s);
  };

  const defaults = defaultsFor(collection);
  const custom =
    settings.budget !== defaults.budget ||
    settings.tau !== defaults.tau ||
    settings.expansion !== defaults.expansion ||
    Object.values(settings.features).some(Boolean) ||
    settings.domains.length > 0;
  const adaptive = mode === "auto";
  const finalConf = resp ? [...resp.trace].reverse().find((s) => s.confidence != null)?.confidence ?? null : null;
  const route = resp?.route ? parseRoute(resp.route.note) : null;
  const top = resp?.results[0]?.score ?? 0;

  const ranked = collection.costs
    ? Object.entries(collection.costs).filter(([m]) => m !== "bm25_ie").sort((a, b) => a[1] - b[1])
    : [];
  const maxCost = Math.max(...ranked.map(([, c]) => c), 1e-9);

  return (
    <div>
      <form
        role="search"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="panel flex items-center gap-3 py-2 pl-4 pr-2 transition-shadow duration-200 focus-within:border-accent/50 focus-within:shadow-focus">
          <SearchIcon className="h-5 w-5 flex-none text-ink-faint" aria-hidden />
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question or type a few keywords"
            aria-label="Query"
            autoFocus
            className="min-w-0 flex-1 border-none bg-transparent py-2 text-[1.08rem] text-ink outline-none placeholder:text-ink-faint"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="inline-flex h-10 flex-none items-center gap-2 rounded-ctl bg-ink px-4 text-[0.9rem] font-medium text-canvas transition-opacity hover:opacity-90 active:opacity-80 disabled:opacity-35"
          >
            {loading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <SearchIcon className="h-4 w-4 sm:hidden" />}
            <span className="max-sm:sr-only">Search</span>
          </button>
        </div>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <label className="relative">
          <select
            aria-label="Method"
            value={mode}
            onChange={(e) => changeMode(e.target.value as Mode)}
            className="h-[2.05rem] cursor-pointer appearance-none rounded-ctl border border-line/[0.12] bg-surface pl-3 pr-8 text-[0.84rem] font-medium text-ink outline-none focus:border-accent/60 focus:shadow-focus"
          >
            <option value="auto">{methodLabel("auto")} (recommended)</option>
            <optgroup label="Fixed methods">
              {[...PRIMARY.slice(1), ...OTHER].map((m) => (
                <option key={m} value={m}>
                  {methodLabel(m)}
                </option>
              ))}
            </optgroup>
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
        </label>
        <span className="text-[0.82rem] text-ink-faint max-md:w-full">{MODE_HINT[mode]}</span>
        <button
          type="button"
          onClick={() => setShowSettings((v) => !v)}
          aria-expanded={showSettings}
          className={clsx(
            "ml-auto inline-flex h-[2.05rem] items-center gap-2 rounded-ctl border px-3 text-[0.8rem] font-medium transition-colors",
            showSettings || custom ? "border-accent/40 bg-accent-soft text-accent-ink" : "border-line/[0.12] bg-surface text-ink-soft hover:text-ink",
          )}
        >
          <SlidersHorizontal className="h-3.5 w-3.5" />
          Settings
          {custom && <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-label="changed from defaults" />}
        </button>
      </div>

      {showSettings && (
        <div className="panel mt-3 grid animate-reveal gap-8 p-5 md:grid-cols-[1fr_1fr_0.9fr]">
          <div className="grid content-start gap-5">
            <label className="grid gap-1.5 text-[0.86rem]">
              <span className="flex justify-between font-medium">
                Budget per query <span className="num text-accent-ink">{fmt(settings.budget, 1)} units</span>
              </span>
              <input
                type="range"
                min={0.5}
                max={collection.max_budget}
                step={0.5}
                value={settings.budget}
                onChange={(e) => setSettings({ ...settings, budget: Number(e.target.value) })}
                onPointerUp={() => changeSettings({})}
                onKeyUp={() => changeSettings({})}
                aria-label="Budget"
              />
              <span className="text-[0.8rem] text-ink-soft">
                1 unit is one BM25 query on this machine. Dense search costs {fmt(collection.costs.dense, 1)}, expansion{" "}
                {fmt(collection.costs.expansion, 1)}.
              </span>
            </label>
            <label className="grid gap-1.5 text-[0.86rem]">
              <span className="flex justify-between font-medium">
                Confidence threshold τ <span className="num text-accent-ink">{fmt(settings.tau)}</span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={settings.tau}
                disabled={settings.features.risk_controlled_tau}
                onChange={(e) => setSettings({ ...settings, tau: Number(e.target.value) })}
                onPointerUp={() => changeSettings({})}
                onKeyUp={() => changeSettings({})}
                aria-label="Confidence threshold"
              />
              <span className="text-[0.8rem] text-ink-soft">Stop as soon as confidence reaches τ. A higher τ escalates more and costs more.</span>
            </label>
          </div>
          <div className="grid content-start gap-1">
            <div className="label mb-1">Escalation ladder</div>
            <Toggle
              checked={settings.expansion}
              onChange={(v) => changeSettings({ expansion: v })}
              label="Query expansion in the ladder"
              hint="The last and most expensive step: pseudo-relevance feedback with a drift filter"
            />
            <div className="label mb-1 mt-4">ContextWeave+ extensions</div>
            {FEATURES.map((f) => (
              <Toggle
                key={f.id}
                checked={settings.features[f.id]}
                onChange={(v) => changeSettings({ features: { ...settings.features, [f.id]: v } })}
                label={f.label}
                hint={f.hint}
                disabled={!collection.features[f.id]}
                disabledHint="No fitted model for this collection. Run experiments/fit_and_tune.py."
              />
            ))}
          </div>
          <div className="grid content-start gap-4">
            {collection.domains.length > 0 && (
              <div>
                <div className="label mb-2">Only search these domains</div>
                <div className="flex flex-wrap gap-1.5">
                  {collection.domains.map((d) => (
                    <Chip
                      key={d}
                      active={settings.domains.includes(d)}
                      onClick={() =>
                        changeSettings({
                          domains: settings.domains.includes(d) ? settings.domains.filter((x) => x !== d) : [...settings.domains, d],
                        })
                      }
                    >
                      {d}
                    </Chip>
                  ))}
                </div>
              </div>
            )}
            <div className="rounded-ctl bg-raised p-3 text-[0.8rem] text-ink-soft">
              Defaults: {collection.defaults.source}.
              <button
                type="button"
                onClick={() => changeSettings(defaults)}
                disabled={!custom}
                className="mt-2 flex items-center gap-1.5 font-medium text-ink hover:text-accent disabled:text-ink-faint"
              >
                <RotateCcw className="h-3.5 w-3.5" /> Reset to tuned defaults
              </button>
            </div>
          </div>
        </div>
      )}

      {!resp && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-[0.82rem] text-ink-faint">Try</span>
          {collection.examples.map((ex) => (
            <Chip key={ex} onClick={() => submit(ex)}>
              {ex}
            </Chip>
          ))}
        </div>
      )}

      {error && <Callout tone="warn" title="Search failed.">{error}</Callout>}
      {loading && !resp && (
        <div className="mt-8 grid gap-4 lg:grid-cols-[minmax(0,1fr)_23rem]" aria-busy="true">
          <div className="grid gap-3">
            {slow && <p className="text-[0.88rem] text-ink-soft">Loading the indexes and the embedding model. Only the first search on a collection is slow.</p>}
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </div>
          <Skeleton className="h-80 max-lg:hidden" />
        </div>
      )}

      {!resp && !loading && !error && (
        <div className="mt-10 grid gap-10 border-t hairline pt-8 lg:grid-cols-[1fr_1fr]">
          <div>
            <h3 className="text-[1.05rem] font-semibold">How it decides</h3>
            <p className="mt-1.5 max-w-[60ch] text-[0.92rem] leading-relaxed text-ink-soft">
              ContextWeave decides per query how much retrieval effort to spend. A router picks the cheapest suitable method, a
              confidence estimate checks whether the results are already good enough, and the controller escalates only while it is
              unsure and the budget allows.
            </p>
            <ol className="mt-5 grid gap-3 text-[0.9rem]">
              {[
                ["Short, exact queries", "usually stop after one cheap BM25 pass."],
                ["Questions and paraphrases", "are routed to dense, semantic search."],
                ["Uncertain results", "escalate to fusion and query expansion, until τ or the budget is reached."],
              ].map(([a, b], i) => (
                <li key={a} className="grid grid-cols-[1.5rem_1fr] gap-2">
                  <span className="num font-medium text-ink-faint">{i + 1}</span>
                  <span>
                    <b className="font-medium">{a}</b> <span className="text-ink-soft">{b}</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
          {ranked.length > 0 && (
            <div>
              <h3 className="text-[1.05rem] font-semibold">What each step costs</h3>
              <p className="mt-1.5 text-[0.92rem] text-ink-soft">
                Measured on this machine, in units of one BM25 query. The default budget is{" "}
                <b className="num font-medium text-ink">{fmt(defaults.budget, 1)}</b>.
              </p>
              <ul className="mt-5 grid gap-2.5">
                {ranked.map(([m, c]) => (
                  <li key={m} className="grid grid-cols-[minmax(0,10.5rem)_1fr_3.5rem] items-center gap-3 text-[0.86rem]">
                    <MethodTag method={m} className="truncate" />
                    <span className="h-2 overflow-hidden rounded-full bg-sunken">
                      <span className={clsx("block h-full rounded-full", kindBg(m))} style={{ width: `${Math.max((c / maxCost) * 100, 1)}%` }} />
                    </span>
                    <span className="num text-right text-ink-soft">{fmt(c)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {resp && (
        <div className={clsx("mt-6 transition-opacity duration-300", loading && "opacity-60")} aria-busy={loading}>
          <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_23rem]">
            <section aria-label="Results" className="min-w-0">
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <h2 className="text-[1.15rem] font-semibold tracking-[-0.01em]">Results</h2>
                <span className="text-[0.84rem] text-ink-faint">
                  <span className="num">{resp.results.length}</span> passages for{" "}
                  <span className="text-ink-soft">“{resp.query.raw}”</span>
                </span>
              </div>
              {resp.results.length ? (
                <ol className="panel">
                  {resp.results.map((r) => (
                    <ResultCard key={r.passage_id} r={r} topScore={top} collection={collection.id} />
                  ))}
                </ol>
              ) : (
                <Callout tone="warn">
                  No passage matched.{" "}
                  {resp.mode === "boolean"
                    ? "Boolean AND needs every term in the same passage, so try fewer terms."
                    : "Try other words, or the Dense method, which matches by meaning."}
                </Callout>
              )}
            </section>

            <aside className="panel min-w-0 lg:sticky lg:top-[4.25rem]" aria-label="How the controller decided">
              <div className="border-b hairline px-5 py-4">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-[0.98rem] font-semibold">How it decided</h2>
                  {route?.cls && <Badge tone="accent">{route.cls} query</Badge>}
                </div>
                <p className="mt-1 text-[0.86rem] leading-snug text-ink-soft">{stopSentence(resp.stop_reason, resp.mode)}</p>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-4 border-b hairline px-5 py-4">
                <Stat
                  label="Cost"
                  value={fmt(resp.total_cost)}
                  sub={adaptive && resp.budget ? `of ${fmt(resp.budget, 1)}` : "units"}
                  tone={adaptive && resp.budget && resp.total_cost > resp.budget ? "fail" : undefined}
                />
                <Stat
                  label="Confidence"
                  value={fmt(finalConf)}
                  sub={resp.tau != null && (adaptive || mode === "route") ? `τ ${fmt(resp.tau)}` : undefined}
                  note={
                    finalConf == null
                      ? "Only the adaptive modes measure it."
                      : adaptive
                        ? resp.tau_source === "learn-then-test"
                          ? "τ from risk control"
                          : undefined
                        : "measured for reference"
                  }
                  tone={finalConf != null && resp.tau != null && adaptive ? (finalConf >= resp.tau ? "pass" : "fail") : undefined}
                />
                <Stat label="Methods run" value={resp.methods_run.length} />
                <Stat label="Latency" value={ms(resp.latency_ms)} sub="server" />
              </div>
              <div className="border-b hairline px-5 py-4">
                <div className="label mb-1">Budget spent per step</div>
                <SpendLedger trace={resp.trace} tau={resp.tau} budget={resp.budget} mode={resp.mode} total={resp.total_cost} />
              </div>
              <div className="px-5 py-4">
                <div className="label mb-2">Decision path</div>
                <DecisionPath trace={resp.trace} tau={resp.tau} mode={resp.mode} />
              </div>
            </aside>
          </div>

          <div className="mt-10 border-t hairline pt-6">
            <button
              type="button"
              onClick={() => setShowAnalysis((v) => !v)}
              aria-expanded={showAnalysis}
              className="inline-flex items-center gap-1.5 text-[0.9rem] font-medium text-accent-ink hover:text-accent"
            >
              <ChevronDown className={clsx("h-4 w-4 transition-transform duration-200", showAnalysis && "rotate-180")} />
              {showAnalysis ? "Hide the full analysis" : "Show the full analysis"}
            </button>
            {!showAnalysis && (
              <p className="mt-1 text-[0.82rem] text-ink-faint">
                How each query term was read, what every method ranked on its own, and why the controller was or was not confident.
              </p>
            )}
            {showAnalysis && (
              <div className="animate-reveal">
                <section aria-label="How the query was read" className="mt-8">
                  <h2 className="text-[1.15rem] font-semibold tracking-[-0.01em]">How the query was read</h2>
                  <div className="mt-3 max-w-xl">
                    <TermChips resp={resp} />
                  </div>
                </section>
                <QueryInsights resp={resp} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
