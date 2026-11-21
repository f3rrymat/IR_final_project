"""Index Elimination (Manning et al., IIR §7.1.2).

1. Ignore query terms whose IDF (ln N/df) is below ``idf_threshold`` — their
   postings are long and they contribute little to the ranking.
2. Only score passages that contain at least ``m`` of the remaining terms
   (m = min(min_terms_matched, number of remaining terms)).

Registered as ``bm25_ie`` (full postings + elimination). The champion-list
retriever reuses ``eliminate_terms`` so S3 = Champion Lists + Index Elimination.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage
from contextweave.lexical.bm25 import BM25Scorer, to_results


def eliminate_terms(scorer: BM25Scorer, weights: dict[str, float], idf_threshold: float) -> tuple[dict[str, float], list[str]]:
    """Return (kept weights, dropped terms). Unknown terms are dropped; if every
    known term falls below the threshold, the single highest-IDF term is kept."""
    idx = scorer.index
    known = {t: w for t, w in weights.items() if idx.df(t) > 0}
    kept = {t: w for t, w in known.items() if idx.idf(t) >= idf_threshold}
    if not kept and known:
        best = max(known, key=idx.idf)
        kept = {best: known[best]}
    dropped = [t for t in weights if t not in kept]
    return kept, dropped


def min_match_filter(scorer: BM25Scorer, terms: list[str], candidates: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidate passage indices containing >= m of ``terms``.
    Returns (filtered candidates, match counts for the filtered candidates)."""
    counts = np.zeros(candidates.shape[0], dtype=np.int32)
    for t in terms:
        p = scorer.index.get(t)
        if p is None:
            continue
        j = np.searchsorted(p.docs, candidates)
        j = np.minimum(j, p.df - 1)
        counts += (p.docs[j] == candidates).astype(np.int32)
    keep = counts >= m
    return candidates[keep], counts[keep]


class IndexEliminationRetriever:
    name = "bm25_ie"

    def __init__(self, scorer: BM25Scorer, idf_threshold: float = 1.5, min_terms_matched: int = 2,
                 cost: float = 0.6):
        self.scorer, self.idf_threshold, self.min_terms = scorer, idf_threshold, min_terms_matched
        self.cost = cost
        self.last_stats: dict = {}

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        with timed() as t:
            weights = self.scorer.query_weights(query)
            kept, dropped = eliminate_terms(self.scorer, weights, self.idf_threshold)
            scores, touched = self.scorer.accumulate(kept, self.scorer.index.mask(candidate_ids))
            cand = np.flatnonzero(scores > 0)
            m = min(self.min_terms, len(kept))
            filt, _ = min_match_filter(self.scorer, list(kept), cand, m)
            if filt.size < k and m > 1:  # relax rather than return too little
                filt, m = cand, 1
            valid = np.zeros(self.scorer.index.N, dtype=bool)
            valid[filt] = True
            res = to_results(self.scorer.index, scores, k, self.name, valid=valid)
            pos = self.scorer.index.pos
            for r in res[:10]:  # explain the top 10 only, like bm25
                r.explanation = {"terms": self.scorer.explain(kept, pos[r.passage_id]), "eliminated": dropped}
        self.last_stats = {"postings_touched": touched, "passages_scored": int(filt.size), "kept_terms": list(kept),
                           "eliminated_terms": dropped, "min_match": m, "latency_ms": t.ms, "N": self.scorer.index.N}
        return res
