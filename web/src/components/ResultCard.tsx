import clsx from "clsx";
import { Check, ChevronDown, Minus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, type Passage, type Result } from "../lib/api";
import { fmt, methodKind, methodLabel, pct } from "../lib/format";
import { Badge, Meter, MethodTag, Skeleton } from "./ui";

export function Snippet({ r }: { r: Result }) {
  const s = r.snippet;
  return (
    <p className="snippet prose-passage mt-2 text-[1rem] text-ink/90">
      {s.before && "… "}
      {s.segments.map(([text, hit], i) => (hit ? <mark key={i}>{text}</mark> : <span key={i}>{text}</span>))}
      {s.after && " …"}
    </p>
  );
}

function Found({ on, label }: { on: boolean | null; label: string }) {
  if (on == null) return null;
  return (
    <span className={clsx("inline-flex items-center gap-1", on ? "text-ink-soft" : "text-ink-faint line-through decoration-ink-faint/50")}>
      {on ? <Check className="h-3 w-3 text-pass" /> : <Minus className="h-3 w-3" />}
      {label}
    </span>
  );
}

/** How the fused score was put together: α·lexical and (1−α)·semantic as two blocks of one bar. */
function FusionBar({ alpha, lex, sem }: { alpha: number; lex: number; sem: number }) {
  const a = alpha * lex;
  const b = (1 - alpha) * sem;
  return (
    <div>
      <div className="flex h-3 overflow-hidden rounded-full bg-sunken" role="img" aria-label={`lexical part ${fmt(a)}, semantic part ${fmt(b)}`}>
        <span className="bg-lexical" style={{ width: `${a * 100}%` }} />
        <span className="bg-semantic" style={{ width: `${b * 100}%` }} />
      </div>
      <div className="num mt-1.5 flex flex-wrap justify-between gap-x-4 text-[0.78rem] text-ink-soft">
        <span>
          <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-lexical" />
          lexical {fmt(alpha)} × {fmt(lex)} = <b className="font-medium text-ink">{fmt(a)}</b>
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-semantic" />
          semantic {fmt(1 - alpha)} × {fmt(sem)} = <b className="font-medium text-ink">{fmt(b)}</b>
        </span>
      </div>
    </div>
  );
}

function Why({ r }: { r: Result }) {
  const w = r.why;
  const fused = w.alpha != null;
  const maxTerm = Math.max(...w.terms.map((t) => t.score ?? 0), 1e-9);
  return (
    <div className="mt-3 grid gap-6 rounded-ctl border hairline bg-raised p-4 md:grid-cols-2">
      <div className="min-w-0">
        <div className="label mb-2">{fused ? "Fused score" : "Score"}</div>
        {fused ? (
          <>
            <FusionBar alpha={w.alpha ?? 0} lex={w.lexical_norm ?? 0} sem={w.semantic_norm ?? 0} />
            <p className="mt-3 text-[0.8rem] leading-relaxed text-ink-soft">
              score = α·lexical + (1−α)·semantic, both min-max normalised, with α = <b className="num font-medium text-ink">{fmt(w.alpha)}</b>.
              {w.lexical_method && <> The lexical list came from {methodLabel(w.lexical_method)}.</>}
            </p>
            <div className="mt-2 flex flex-wrap gap-x-4 text-[0.8rem]">
              <Found on={w.in_lexical} label="in the lexical list" />
              <Found on={w.in_semantic} label="in the semantic list" />
            </div>
          </>
        ) : (
          <p className="text-[0.86rem] text-ink-soft">
            {methodLabel(r.method)} score <b className="num font-medium text-ink">{fmt(r.score, 4)}</b>
          </p>
        )}
        {w.cosine != null && (
          <div className="mt-4">
            <div className="flex items-baseline justify-between text-[0.8rem] text-ink-soft">
              <span>Cosine similarity to the query embedding</span>
              <b className="num font-medium text-ink">{fmt(w.cosine, 3)}</b>
            </div>
            <Meter value={w.cosine} tone="semantic" className="mt-1.5" label="cosine similarity" />
          </div>
        )}
      </div>
      <div className="min-w-0">
        <div className="label mb-2">Query terms in this passage</div>
        {w.terms.length ? (
          <ul className="grid gap-2">
            {w.terms.map((t) => (
              <li key={t.term} className="grid grid-cols-[minmax(4rem,7rem)_1fr_auto] items-center gap-3 text-[0.82rem]">
                <code className="truncate text-[0.78rem]">{t.term}</code>
                <Meter value={(t.score ?? 0) / maxTerm} tone="lexical" label={`${t.term} contribution`} />
                <span className="num whitespace-nowrap text-right text-[0.76rem] text-ink-faint">
                  tf {t.tf}, idf {fmt(t.idf)}, <span className="text-ink-soft">{pct(t.share)}</span>
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[0.84rem] text-ink-soft">
            No query term occurs in this passage. Dense retrieval found it by meaning, not by matching words.
          </p>
        )}
        {w.eliminated.length > 0 && (
          <p className="mt-3 text-[0.78rem] text-ink-faint">Index elimination ignored: {w.eliminated.join(", ")}</p>
        )}
      </div>
    </div>
  );
}

/** The full passage in a reading sheet that slides in from the right. */
export function PassageDialog({ collection, pid, onClose }: { collection: string; pid: string; onClose: () => void }) {
  const [p, setP] = useState<Passage | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    let live = true;
    api.passage(collection, pid).then(
      (x) => live && setP(x),
      (e) => live && setErr(String(e.message ?? e)),
    );
    return () => {
      live = false;
    };
  }, [collection, pid]);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCloseRef.current();
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      opener?.focus?.(); // return focus to the button that opened the sheet
    };
  }, []);
  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end bg-[rgb(var(--shadow)/0.35)] backdrop-blur-[1px]" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Full passage"
        className="flex h-full w-full max-w-[44rem] animate-reveal flex-col border-l hairline bg-surface shadow-lift"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b hairline px-6 py-4">
          <div className="min-w-0">
            <div className="label">Full passage</div>
            <h3 className="mt-0.5 text-[1.1rem] font-semibold leading-snug">{p?.title ?? "Loading…"}</h3>
          </div>
          <button ref={closeRef} onClick={onClose} aria-label="Close" className="-mr-2 rounded-ctl p-2 text-ink-soft hover:bg-sunken hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {err && <p className="text-fail">{err}</p>}
          {!p && !err && <Skeleton className="h-48" />}
          {p && (
            <>
              <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-[0.82rem]">
                {p.page ? <><dt className="text-ink-faint">Page</dt><dd className="num">{p.page}</dd></> : null}
                <dt className="text-ink-faint">Domain</dt>
                <dd>{p.domain}</dd>
                <dt className="text-ink-faint">Length</dt>
                <dd className="num">{p.n_tokens} words</dd>
                <dt className="text-ink-faint">Passage</dt>
                <dd className="font-mono text-[0.78rem]">{p.passage_id}</dd>
              </dl>
              <p className="prose-passage mt-6 max-w-[66ch] whitespace-pre-line">{p.text}</p>
              <p className="mt-8 break-all border-t hairline pt-4 text-[0.78rem] text-ink-faint">Source: {p.source}</p>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}

export function ResultCard({ r, topScore, collection }: { r: Result; topScore: number; collection: string }) {
  const [open, setOpen] = useState(false);
  const [reading, setReading] = useState(false);
  const k = methodKind(r.method);
  const rel = topScore > 0 ? r.score / topScore : 0;
  return (
    <li className="group border-b hairline px-4 py-5 last:border-b-0 sm:px-6">
      <div className="grid grid-cols-[1.75rem_1fr] gap-x-3 sm:grid-cols-[2.25rem_1fr_7.5rem] sm:gap-x-4">
        <span className={clsx("num pt-px text-[1.05rem] font-semibold", r.rank <= 3 ? "text-ink" : "text-ink-faint")}>{r.rank}</span>
        <div className="min-w-0">
          <h3 className="text-[1.02rem] font-semibold leading-snug tracking-[-0.005em]">
            <button
              type="button"
              onClick={() => setReading(true)}
              title="Read the full passage"
              className="text-left underline-offset-[3px] hover:text-accent-ink hover:underline"
            >
              {r.title}
            </button>
          </h3>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.8rem] text-ink-faint">
            <Badge tone={k === "semantic" ? "semantic" : k === "expand" ? "expand" : k === "fused" ? "plain" : "accent"}>
              <MethodTag method={r.method} />
            </Badge>
            {r.domain && <span>{r.domain}</span>}
            {r.page ? <span className="num">p. {r.page}</span> : null}
            <span className="font-mono text-[0.74rem]">{r.passage_id}</span>
            {r.snippet.hits > 0 && <span className="num">{r.snippet.hits} term hits</span>}
          </div>
        </div>
        <div className="col-start-2 mt-2 sm:col-start-3 sm:row-start-1 sm:mt-0.5" title={`${Math.round(rel * 100)}% of the top score`}>
          <div className="num text-[0.92rem] font-medium sm:text-right">{fmt(r.score, 3)}</div>
        </div>
        <div className="col-start-2 min-w-0 sm:col-end-4">
          <Snippet r={r} />
          <div className="mt-2.5 text-[0.84rem] font-medium">
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              className="inline-flex items-center gap-1 text-accent-ink hover:text-accent"
            >
              <ChevronDown className={clsx("h-4 w-4 transition-transform duration-200", open && "rotate-180")} />
              Why this result?
            </button>
          </div>
          {open && <Why r={r} />}
        </div>
      </div>
      {reading && <PassageDialog collection={collection} pid={r.passage_id} onClose={() => setReading(false)} />}
    </li>
  );
}
