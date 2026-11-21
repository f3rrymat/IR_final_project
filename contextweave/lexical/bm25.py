"""BM25 (Okapi), implemented from scratch. Registered as ``bm25``.

    score(q, d) = sum_t w_q(t) * idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

IDF variants:
  * ``lucene`` (default): idf = ln(1 + (N - df + 0.5) / (df + 0.5)), always > 0.
  * ``okapi``: idf = ln((N - df + 0.5) / (df + 0.5)), negative values replaced by
    eps * mean idf (eps = 0.25). This is exactly what rank-bm25's BM25Okapi does,
    so the test can compare scores one-to-one.

Per-term contribution arrays over each postings list are query independent, so
they are computed once per term and cached (an "impact" precomputation).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from contextweave.core.retriever import finalize
from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage
from contextweave.indexing.inverted_index import InvertedIndex


class BM25Scorer:
    """Shared BM25 maths used by bm25, bm25_champion, index elimination and expansion."""

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75, idf_variant: str = "lucene"):
        self.index, self.k1, self.b, self.idf_variant = index, k1, b, idf_variant
        self._impact: dict[str, np.ndarray] = {}
        self._norm = k1 * (1 - b + b * index.lengths.astype(np.float64) / max(index.avgdl, 1e-9))
        self._mean_idf: Optional[float] = None

    def idf(self, term: str) -> float:
        df, N = self.index.df(term), self.index.N  # noqa: N806
        if df == 0:
            return 0.0
        if self.idf_variant == "lucene":
            return math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        raw = math.log((N - df + 0.5) / (df + 0.5))
        if raw < 0:
            return 0.25 * self.mean_idf()
        return raw

    def mean_idf(self) -> float:
        if self._mean_idf is None:
            N = self.index.N  # noqa: N806
            vals = [math.log((N - p.df + 0.5) / (p.df + 0.5)) for p in self.index.postings.values()]
            self._mean_idf = sum(vals) / max(len(vals), 1)
        return self._mean_idf

    def impact(self, term: str) -> Optional[np.ndarray]:
        """BM25 contribution of ``term`` for every passage in its postings list."""
        if term in self._impact:
            return self._impact[term]
        p = self.index.get(term)
        if p is None:
            return None
        tf = p.tfs.astype(np.float64)
        imp = self.idf(term) * tf * (self.k1 + 1) / (tf + self._norm[p.docs])
        self._impact[term] = imp
        return imp

    def query_weights(self, query: Query) -> dict[str, float]:
        """Query term weights: term counts (like rank-bm25) plus expansion weights."""
        w: dict[str, float] = {}
        for t in query.terms:
            w[t] = w.get(t, 0.0) + 1.0
        for t, x in query.term_weights.items():
            w[t] = w.get(t, 0.0) + x
        return w

    def accumulate(self, weights: dict[str, float], mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
        """Term-at-a-time accumulation. Returns (score array, postings touched)."""
        scores = np.zeros(self.index.N, dtype=np.float64)
        touched = 0
        for term, wq in weights.items():
            p = self.index.get(term)
            if p is None:
                continue
            imp = self.impact(term)
            if mask is None:
                scores[p.docs] += wq * imp  # docs are unique within a postings list
                touched += p.df
            else:
                sel = mask[p.docs]
                scores[p.docs[sel]] += wq * imp[sel]
                touched += int(sel.sum())
        return scores, touched

    def explain(self, weights: dict[str, float], idx: int) -> dict[str, dict[str, float]]:
        """Per-term contribution for one passage (only terms present in it)."""
        out = {}
        for term, wq in weights.items():
            p = self.index.get(term)
            if p is None:
                continue
            j = int(np.searchsorted(p.docs, idx))
            if j < p.df and int(p.docs[j]) == idx:
                out[term] = {"tf": int(p.tfs[j]), "idf": round(self.idf(term), 4),
                             "weight": round(wq, 3), "score": round(float(wq * self.impact(term)[j]), 4)}
        return out


def to_results(index: InvertedIndex, scores: np.ndarray, k: int, method: str,
               valid: Optional[np.ndarray] = None) -> list[ScoredPassage]:
    """Top-k of a dense score array (only passages with score > 0 or marked valid)."""
    cand = np.flatnonzero(scores > 0) if valid is None else np.flatnonzero(valid)
    if cand.size == 0:
        return []
    if cand.size > k:
        part = cand[np.argpartition(-scores[cand], k - 1)[:k]]
    else:
        part = cand
    res = [ScoredPassage(index.passage_ids[i], index.doc_ids[i], float(scores[i]), method) for i in part]
    return finalize(res, k)


class BM25Retriever:
    """Full BM25 over the whole index. Registered as ``bm25``."""

    name = "bm25"

    def __init__(self, scorer: BM25Scorer, cost: float = 1.0, explain_top: int = 10):
        self.scorer, self.cost, self.explain_top = scorer, cost, explain_top
        self.last_stats: dict = {}

    def mask_for(self, candidate_ids: Optional[set[str]]) -> Optional[np.ndarray]:
        return self.scorer.index.mask(candidate_ids)

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        with timed() as t:
            weights = self.scorer.query_weights(query)
            scores, touched = self.scorer.accumulate(weights, self.mask_for(candidate_ids))
            res = to_results(self.scorer.index, scores, k, self.name)
            pos = self.scorer.index.pos
            for r in res[: self.explain_top]:
                r.explanation = {"terms": self.scorer.explain(weights, pos[r.passage_id])}
        self.last_stats = {"postings_touched": touched, "passages_scored": int((scores > 0).sum()),
                           "latency_ms": t.ms, "N": self.scorer.index.N}
        return res
