import { BarChart3, ClipboardCheck, Moon, Search, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Callout, Segmented, Skeleton } from "./components/ui";
import { api, type Collection, type CollectionsResponse } from "./lib/api";
import { compact } from "./lib/format";
import { EvaluationPage } from "./pages/EvaluationPage";
import { JudgePage } from "./pages/JudgePage";
import { SearchPage } from "./pages/SearchPage";

export type Page = "search" | "evaluation" | "judge";
const PATHS: Record<Page, string> = { search: "/", evaluation: "/evaluation", judge: "/judge" };
const INTRO: Record<Page, { title: string; text: string }> = {
  search: {
    title: "Search",
    text: "ContextWeave spends retrieval effort only where a query needs it, and shows every decision it made.",
  },
  evaluation: {
    title: "Evaluation",
    text: "How the adaptive controller compares with every fixed method, on queries it never saw while tuning.",
  },
  judge: {
    title: "Relevance judging",
    text: "Grade pooled passages 0, 1 or 2 to build the relevance labels for the CS benchmark.",
  },
};

const pageFromPath = (p: string): Page => (p.startsWith("/evaluation") ? "evaluation" : p.startsWith("/judge") ? "judge" : "search");

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7 flex-none" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#16213a" />
      <path d="M6 11c4.5 0 5.5 10 10 10s5.5-10 10-10" fill="none" stroke="#7e94ff" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M6 21c4.5 0 5.5-10 10-10s5.5 10 10 10" fill="none" stroke="#b49cff" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("cw.theme", theme);
    } catch {
      /* private mode */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))] as const;
}

export default function App() {
  const [meta, setMeta] = useState<CollectionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>(() => pageFromPath(window.location.pathname));
  const [cid, setCid] = useState<string | null>(() => new URLSearchParams(window.location.search).get("c"));
  const [theme, toggleTheme] = useTheme();

  useEffect(() => {
    api.collections().then(
      (d) => {
        setMeta(d);
        setCid((c) => (c && d.collections.some((x) => x.id === c && x.ready) ? c : d.default));
      },
      (e) => setError(e.message),
    );
    const onPop = () => {
      setPage(pageFromPath(window.location.pathname));
      const c = new URLSearchParams(window.location.search).get("c");
      if (c) setCid(c);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = (p: Page, c = cid) => {
    const url = new URL(window.location.href);
    url.pathname = PATHS[p];
    const keep = p === "search" && p === page ? url.searchParams : new URLSearchParams();
    if (c) keep.set("c", c);
    url.search = keep.toString();
    window.history.pushState(null, "", url);
    setPage(p);
    setCid(c);
  };

  const collection: Collection | undefined = meta?.collections.find((c) => c.id === cid);
  const ready = meta?.collections.filter((c) => c.ready) ?? [];
  const intro = INTRO[page];

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b hairline bg-canvas/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1320px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 sm:px-6">
          <a
            href="/"
            onClick={(e) => {
              e.preventDefault();
              navigate("search");
            }}
            className="flex items-center gap-2.5 rounded-ctl"
          >
            <Logo />
            <h1 className="text-[1.05rem] font-semibold tracking-[-0.01em]">ContextWeave</h1>
          </a>
          <nav aria-label="Pages" className="order-3 w-full sm:order-none sm:w-auto">
            <Segmented
              label="Pages"
              value={page}
              onChange={(p) => navigate(p)}
              options={[
                { value: "search", label: <><Search className="h-4 w-4" />Search</> },
                { value: "evaluation", label: <><BarChart3 className="h-4 w-4" />Evaluation</> },
                { value: "judge", label: <><ClipboardCheck className="h-4 w-4" />Judge</> },
              ]}
            />
          </nav>
          <div className="ml-auto flex items-center gap-2">
            {meta && collection && (
              <Segmented
                label="Collection"
                size="sm"
                value={collection.id}
                onChange={(c) => navigate(page, c)}
                options={meta.collections.map((c) => ({
                  value: c.id,
                  label: c.short,
                  disabled: !c.ready,
                  title: c.ready ? c.description : `Not built yet: python main.py build-index --collection ${c.id}`,
                }))}
              />
            )}
            <button
              type="button"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Use light theme" : "Use dark theme"}
              title={theme === "dark" ? "Use light theme" : "Use dark theme"}
              className="flex h-8 w-8 items-center justify-center rounded-ctl text-ink-soft transition-colors hover:bg-sunken hover:text-ink"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1320px] px-4 pb-24 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-2 pb-6 pt-8 sm:pt-10">
          <div className="min-w-0 max-w-[60ch]">
            <h2 className="text-[clamp(1.6rem,3vw,2.1rem)] font-semibold leading-tight tracking-[-0.025em]">{intro.title}</h2>
            <p className="mt-1.5 text-[0.95rem] text-ink-soft">{intro.text}</p>
          </div>
          {collection && (
            <p className="max-w-[46ch] text-[0.84rem] text-ink-faint sm:text-right">
              <span className="font-medium text-ink">{collection.label}</span>
              {collection.passages != null && <span className="num">, {compact(collection.passages)} passages</span>}
              <span className="block">{collection.description.split(" - ")[1] ?? collection.description}</span>
            </p>
          )}
        </div>

        <main>
          {error && (
            <Callout tone="warn" title="Cannot reach the ContextWeave API.">
              {error}. Start it with <code>python main.py serve</code>.
            </Callout>
          )}
          {!meta && !error && <Skeleton className="h-16" />}
          {meta && ready.length === 0 && (
            <Callout tone="warn" title="No collection is built yet.">
              Run <code>python scripts/get_beir.py</code>, <code>python main.py data</code> and{" "}
              <code>python main.py build-index --collection scifact,nfcorpus</code>, then reload.
            </Callout>
          )}
          {collection && collection.ready && page === "search" && <SearchPage key="search" collection={collection} />}
          {collection && page === "evaluation" && <EvaluationPage collection={collection} />}
          {collection && page === "judge" && <JudgePage collection={collection} />}
        </main>

        <footer className="mt-20 flex flex-wrap justify-between gap-3 border-t hairline pt-5 text-[0.8rem] text-ink-faint">
          <span>
            Inverted index, BM25, TF-IDF, champion lists, fusion and every metric written from scratch. Dense retrieval with
            all-MiniLM-L6-v2 and FAISS.
          </span>
          <a className="underline decoration-ink-faint/40 underline-offset-2 hover:text-ink" href="/api/docs">
            API reference
          </a>
        </footer>
      </div>
    </div>
  );
}
