"""Turn engine objects (SearchResponse, saved result files) into plain JSON for the web UI.

Nothing here computes a retrieval number: everything comes from the engine or from
``experiments/results/*.json``. Keep this module free of FastAPI so it is easy to test.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from contextweave.core.types import SearchResponse
from contextweave.indexing.snippets import snippet_segments
from contextweave.lexical.explain import term_rows

LEXICAL = ("bm25", "bm25_champion", "bm25_ie", "tfidf", "boolean", "expansion")
SEMANTIC = ("dense",)
LABELS = {"bm25": "BM25", "bm25_champion": "BM25 + Champion lists + Index elimination", "bm25_ie": "BM25 + Index elimination",
          "tfidf": "TF-IDF", "boolean": "Boolean", "dense": "Dense (MiniLM + FAISS)", "expansion": "BM25 + query expansion",
          "hybrid": "Fixed hybrid", "fusion": "Fused (lexical + semantic)"}


def _num(x: Any, nd: int = 4) -> Any:
    """Round floats; NaN/inf -> None so the payload is valid JSON."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int,)):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def _signals(d: dict) -> dict:
    return {k: _num(v) for k, v in (d or {}).items() if isinstance(v, (int, float, bool)) or v is None}


def _passage(system, pid: str) -> dict:
    p = system.store[pid]
    return {"passage_id": pid, "doc_id": p.doc_id, "title": p.title, "page": p.page, "domain": p.domain,
            "source": p.source}


def _terms(r) -> list[dict]:
    return [{"term": t["term"], "tf": t.get("tf"), "idf": _num(t.get("idf")), "weight": _num(t.get("weight")),
             "score": _num(t.get("score")), "share": _num(t.get("share"))} for t in term_rows(r)]


def search_payload(resp: SearchResponse, system, highlight_terms: Optional[set[str]] = None) -> dict:
    """Everything the Search page shows for one query."""
    ex = resp.extras
    stats = ex.get("stats", {})
    added = [a for st in stats.values() for a in st.get("added_terms", [])]
    hl = set(resp.query.terms) | {a["term"] for a in added} | (highlight_terms or set())

    results = []
    for r in resp.results:
        p = system.store[r.passage_id]
        e = r.explanation or {}
        results.append({
            **_passage(system, r.passage_id), "rank": r.rank, "score": _num(r.score), "method": r.method,
            "snippet": snippet_segments(p.text, hl, system.normalize, window=45),
            "why": {"alpha": _num(e.get("alpha")), "lexical_method": e.get("lexical_method"),
                    "lexical_norm": _num(e.get("lexical_norm")), "semantic_norm": _num(e.get("semantic_norm")),
                    "cosine": _num(e.get("cosine")), "in_lexical": e.get("in_lexical"),
                    "in_semantic": e.get("in_semantic"), "terms": _terms(r),
                    "eliminated": e.get("eliminated") or []},
        })

    lists = {}
    for m, lst in ex.get("lists", {}).items():
        st = stats.get(m, {})
        lists[m] = {"label": LABELS.get(m, m), "kind": "semantic" if m in SEMANTIC else "lexical",
                    "stats": {k: _num(v) for k, v in st.items() if isinstance(v, (int, float))},
                    "eliminated_terms": st.get("eliminated_terms") or [],
                    "top": [{**_passage(system, r.passage_id), "rank": r.rank, "score": _num(r.score),
                             "terms": _terms(r)[:4]} for r in lst[:10]]}

    # top-10 overlap of the semantic list with the best lexical list that ran
    agreement = None
    lex = next((m for m in ("expansion", "bm25", "bm25_champion", "bm25_ie", "tfidf") if m in lists), None)
    if "dense" in lists and lex:
        a = {x["passage_id"] for x in lists["dense"]["top"]}
        b = {x["passage_id"] for x in lists[lex]["top"]}
        agreement = {"lexical": lex, "overlap": len(a & b), "of": min(10, len(a), len(b))}

    exp = stats.get("expansion")
    idx, sc = system.index, system.scorer
    terms = [{"term": t, "df": idx.df(t), "idf": _num(idx.idf(t), 3), "bm25_idf": _num(sc.idf(t), 3),
              "known": idx.df(t) > 0} for t in dict.fromkeys(resp.query.terms)]
    route = next((s for s in resp.trace if s.action == "route"), None)
    return {
        "query": {"raw": resp.query.raw, "terms": terms},
        "mode": resp.mode, "total_cost": _num(resp.total_cost, 3), "latency_ms": _num(resp.total_latency_ms, 2),
        "budget": _num(ex.get("budget"), 3), "tau": _num(ex.get("tau"), 3), "tau_source": ex.get("tau_source"),
        "stop_reason": ex.get("stop_reason") or "", "methods_run": ex.get("methods_run", []),
        "route": None if route is None else {"method": route.method, "note": route.note,
                                             "features": _signals(route.signals)},
        "trace": [{"action": s.action, "method": s.method, "cost": _num(s.cost, 3),
                   "budget_left": _num(s.budget_left, 3), "confidence": _num(s.confidence, 3),
                   "signals": _signals(s.signals), "latency_ms": _num(s.latency_ms, 2), "note": s.note}
                  for s in resp.trace],
        "results": results, "lists": lists, "agreement": agreement,
        "expansion": None if exp is None else {"added_terms": [{k: _num(v) for k, v in a.items()} for a in exp.get("added_terms", [])],
                                               "feedback": exp.get("feedback", [])},
    }


def evaluation_payload(d: dict) -> dict:
    """Compact view of one ``eval_<collection>_<split>_latest.json`` (per-query rows dropped)."""
    def row(sid: str, r: dict) -> dict:
        a = r.get("aggregate") or {}
        return {"id": sid, "label": r.get("label", sid),
                "metrics": {k: _num(a.get(k)) for k in ("P@5", "Recall@10", "MAP", "NDCG@10", "MRR")} if a else None,
                "mean_cost": _num(r.get("mean_cost"), 3), "escalation_rate": _num(r.get("escalation_rate"), 3),
                "latency_median_ms": _num(r.get("latency_median_ms"), 2),
                "latency_p95_ms": _num(r.get("latency_p95_ms"), 2),
                "by_type": {t: {"NDCG@10": _num(v.get("NDCG@10")), "n": v.get("n_queries"),
                                "mean_cost": _num(v.get("mean_cost"), 3)} for t, v in (r.get("by_type") or {}).items()},
                "tau": _num((r.get("spec") or {}).get("tau"), 3), "budget": _num((r.get("spec") or {}).get("budget"), 3)}

    systems = [row(k, v) for k, v in d.get("systems", {}).items()]
    sig = d.get("significance") or {}
    significance = [{"a": a, "b": b, "mean_diff": _num(v.get("mean_diff")), "p_holm": _num(v.get("t_p_holm"), 6),
                     "wilcoxon_p": _num(v.get("wilcoxon_p"), 6), "ci95": [_num(x) for x in v.get("ci95", [])]}
                    for b, comp in sig.items() for a, v in comp.items()]
    hw = d.get("hardware") or {}
    return {
        "collection": d.get("collection"), "split": d.get("split"), "n_queries": d.get("n_queries"),
        "timestamp_utc": d.get("timestamp_utc"), "labelled": any(s["metrics"] for s in systems),
        "hardware": {"cpu": hw.get("cpu"), "cpu_count": hw.get("cpu_count"), "platform": hw.get("platform")},
        "systems": systems,
        "ablations": [row(k, v) for k, v in (d.get("ablations") or {}).items()],
        "sweep": [{k: _num(v) for k, v in p.items()} for p in d.get("sweep_S8") or []],
        "risk_curve": [{k: _num(v) for k, v in p.items()} for p in d.get("risk_curve") or []],
        "significance": significance,
    }
