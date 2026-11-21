"""Champion Lists (+ Index Elimination). Registered as ``bm25_champion``.

Offline: for every term keep the ``r`` postings with the highest BM25 term
contribution (its "champions").
Query time: apply Index Elimination to the query terms, take the union of the
remaining terms' champion lists as the candidate set, keep candidates matching
at least m terms, and score only those candidates with BM25 (Manning et al.,
IIR §7.1.3–7.1.4). Everything outside the candidate set is never scored.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage
from contextweave.lexical.bm25 import BM25Scorer, to_results
from contextweave.lexical.index_elimination import eliminate_terms, min_match_filter


class ChampionRetriever:
    name = "bm25_champion"

    def __init__(self, scorer: BM25Scorer, r: int = 50, idf_threshold: float = 1.5,
                 min_terms_matched: int = 2, cost: float = 0.3):
        self.scorer, self.r, self.cost = scorer, r, cost
        self.idf_threshold, self.min_terms = idf_threshold, min_terms_matched
        self.champions: dict[str, np.ndarray] = {}
        self.last_stats: dict = {}
        self._build()

    def _build(self) -> None:
        for term, p in self.scorer.index.postings.items():
            if p.df <= self.r:
                self.champions[term] = p.docs
            else:
                imp = self.scorer.impact(term)
                top = np.argpartition(-imp, self.r - 1)[: self.r]
                self.champions[term] = np.sort(p.docs[top])

    def _score_candidates(self, weights: dict[str, float], cand: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.scorer.index.N, dtype=np.float64)
        for t, wq in weights.items():
            p = self.scorer.index.get(t)
            if p is None:
                continue
            j = np.minimum(np.searchsorted(p.docs, cand), p.df - 1)
            hit = p.docs[j] == cand
            scores[cand[hit]] += wq * self.scorer.impact(t)[j[hit]]
        return scores

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        with timed() as t:
            weights = self.scorer.query_weights(query)
            kept, dropped = eliminate_terms(self.scorer, weights, self.idf_threshold)
            lists = [self.champions[t] for t in kept if t in self.champions]
            cand = np.unique(np.concatenate(lists)) if lists else np.zeros(0, dtype=np.int32)
            touched = int(sum(len(x) for x in lists))
            mask = self.scorer.index.mask(candidate_ids)
            if mask is not None:
                cand = cand[mask[cand]]
            m = min(self.min_terms, len(kept))
            filt, _ = min_match_filter(self.scorer, list(kept), cand, m)
            if filt.size < k and m > 1:
                filt, m = cand, 1
            scores = self._score_candidates(kept, filt)
            valid = np.zeros(self.scorer.index.N, dtype=bool)
            valid[filt] = True
            res = to_results(self.scorer.index, scores, k, self.name, valid=valid)
            pos = self.scorer.index.pos
            for r in res[:10]:  # explain the top 10 only, like bm25
                r.explanation = {"terms": self.scorer.explain(kept, pos[r.passage_id]), "eliminated": dropped,
                                 "champion_r": self.r}
        N = self.scorer.index.N  # noqa: N806
        self.last_stats = {"postings_touched": touched, "passages_scored": int(filt.size),
                           "share_not_scored": 1.0 - filt.size / max(N, 1), "kept_terms": list(kept),
                           "eliminated_terms": dropped, "min_match": m, "latency_ms": t.ms, "N": N}
        return res
