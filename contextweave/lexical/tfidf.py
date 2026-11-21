"""TF-IDF vector space model with cosine similarity. Registered as ``tfidf``.

    w(t, d) = (1 + ln tf(t, d)) * ln(N / df(t))        (lnc-style log tf, idf on both sides: ltc.ltc)
    cos(q, d) = sum_t w(t, q) w(t, d) / (|q| |d|)

Document vector lengths are precomputed once, so cosine length normalisation
costs nothing at query time.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage
from contextweave.indexing.inverted_index import InvertedIndex
from contextweave.lexical.bm25 import to_results


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, index: InvertedIndex, cost: float = 1.0, explain_top: int = 10):
        self.index, self.cost, self.explain_top = index, cost, explain_top
        self._w: dict[str, np.ndarray] = {}
        norms2 = np.zeros(index.N, dtype=np.float64)
        for term, p in index.postings.items():
            w = self._weights(term)
            norms2[p.docs] += w * w
        self.doc_norm = np.sqrt(norms2)
        self.doc_norm[self.doc_norm == 0] = 1.0
        self.last_stats: dict = {}

    def _weights(self, term: str) -> np.ndarray:
        if term not in self._w:
            p = self.index.postings[term]
            self._w[term] = (1.0 + np.log(p.tfs.astype(np.float64))) * self.index.idf(term)
        return self._w[term]

    def query_vector(self, query: Query) -> dict[str, float]:
        tf: dict[str, float] = {}
        for t in query.terms:
            tf[t] = tf.get(t, 0.0) + 1.0
        vec = {t: (1.0 + math.log(c)) * self.index.idf(t) for t, c in tf.items() if self.index.df(t)}
        for t, x in query.term_weights.items():
            if self.index.df(t):
                vec[t] = vec.get(t, 0.0) + x * self.index.idf(t)
        return vec

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        with timed() as t:
            qv = self.query_vector(query)
            qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
            mask = self.index.mask(candidate_ids)
            scores = np.zeros(self.index.N, dtype=np.float64)
            touched = 0
            for term, wq in qv.items():
                p = self.index.postings[term]
                w = self._weights(term)
                if mask is None:
                    scores[p.docs] += wq * w
                    touched += p.df
                else:
                    sel = mask[p.docs]
                    scores[p.docs[sel]] += wq * w[sel]
                    touched += int(sel.sum())
            scored = int((scores > 0).sum())
            scores /= self.doc_norm * qn
            res = to_results(self.index, scores, k, self.name)
            pos = self.index.pos
            for r in res[: self.explain_top]:
                i = pos[r.passage_id]
                terms = {}
                for term, wq in qv.items():
                    p = self.index.postings[term]
                    j = int(np.searchsorted(p.docs, i))
                    if j < p.df and int(p.docs[j]) == i:
                        terms[term] = {"tf": int(p.tfs[j]), "idf": round(self.index.idf(term), 4),
                                       "score": round(float(wq * self._weights(term)[j] / (self.doc_norm[i] * qn)), 4)}
                r.explanation = {"terms": terms, "doc_norm": round(float(self.doc_norm[i]), 3)}
        self.last_stats = {"postings_touched": touched, "passages_scored": scored, "latency_ms": t.ms,
                           "N": self.index.N}
        return res
