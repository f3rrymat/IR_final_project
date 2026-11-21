"""Score normalisation and weighted fusion (spec §8.5).

    S(q, d) = alpha(q) * L(q, d) + (1 - alpha(q)) * V(q, d)

L is the min-max normalised score of the most complete lexical list available
(expansion > bm25 > bm25_ie > bm25_champion > tfidf), V the normalised dense
score. A passage missing from one list gets 0 for that part.
"""
from __future__ import annotations

from typing import Optional

from contextweave.core.retriever import finalize
from contextweave.core.types import ScoredPassage

LEXICAL_PRIORITY = ["expansion", "bm25", "bm25_ie", "bm25_champion", "tfidf", "boolean"]
SEMANTIC = "dense"


def minmax(results: list[ScoredPassage]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {r.passage_id: 1.0 for r in results}
    return {r.passage_id: (r.score - lo) / (hi - lo) for r in results}


def zscore(results: list[ScoredPassage]) -> dict[str, float]:
    import statistics

    if len(results) < 2:
        return {r.passage_id: 1.0 for r in results}
    s = [r.score for r in results]
    mu, sd = statistics.fmean(s), statistics.pstdev(s) or 1.0
    return {r.passage_id: (r.score - mu) / sd for r in results}


def lexical_key(lists: dict[str, list[ScoredPassage]]) -> Optional[str]:
    for m in LEXICAL_PRIORITY:
        if m in lists:
            return m
    return None


def normalise_single(results: list[ScoredPassage], how: str = "minmax") -> list[ScoredPassage]:
    norm = minmax(results) if how == "minmax" else zscore(results)
    for r in results:
        r.norm_score = norm[r.passage_id]
    return results


def fuse(lists: dict[str, list[ScoredPassage]], alpha: float, k: int, how: str = "minmax") -> list[ScoredPassage]:
    """Fuse the best lexical list with the dense list. With only one list, return it normalised."""
    lk = lexical_key(lists)
    dense = lists.get(SEMANTIC)
    if lk is None and dense is None:
        return []
    if lk is None or dense is None:
        only = lists[lk] if lk else dense
        out = [ScoredPassage(r.passage_id, r.doc_id, r.score, r.method, r.rank, None, dict(r.explanation))
               for r in only]
        return normalise_single(out, how)[:k]
    norm = minmax if how == "minmax" else zscore
    L, V = norm(lists[lk]), norm(dense)  # noqa: N806
    by_id = {r.passage_id: r for r in dense}
    by_id.update({r.passage_id: r for r in lists[lk]})
    lex_expl = {r.passage_id: r.explanation for r in lists[lk]}
    dense_raw = {r.passage_id: r.score for r in dense}
    out = []
    for pid, r in by_id.items():
        l, v = L.get(pid, 0.0), V.get(pid, 0.0)
        s = alpha * l + (1 - alpha) * v
        expl = {"alpha": round(alpha, 3), "lexical_method": lk, "lexical_norm": round(l, 4),
                "semantic_norm": round(v, 4), "cosine": round(dense_raw[pid], 4) if pid in dense_raw else None,
                "in_lexical": pid in L, "in_semantic": pid in V}
        if pid in lex_expl:
            expl.update({kk: vv for kk, vv in lex_expl[pid].items() if kk in ("terms", "added_terms", "eliminated")})
        out.append(ScoredPassage(pid, r.doc_id, s, "fusion", 0, s, expl))
    return finalize(out, k)


def dedupe(results: list[ScoredPassage]) -> list[ScoredPassage]:
    seen, out = set(), []
    for r in results:
        if r.passage_id not in seen:
            seen.add(r.passage_id)
            out.append(r)
    for i, r in enumerate(out, 1):
        r.rank = i
    return out
