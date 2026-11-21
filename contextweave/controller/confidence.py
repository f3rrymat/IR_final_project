"""Confidence estimate on the current top 10 (spec §9), plus optional QPP signals.

Spec signals
  margin         1 - (10th normalised score), i.e. gap between best and 10th
  coverage       mean over the top 5 passages of the IDF-weighted share of query terms they contain
  concentration  share of the top 10 that come from the most frequent document
  agreement      |lexical top-10 ∩ dense top-10| / 10, only when both have run

Novel (flag ``qpp_signals``) — query performance prediction signals adapted to be scale free:
  nqc   coefficient of variation of the lexical top-10 raw scores
        (NQC, Shtok et al. 2012, with the corpus score replaced by the list mean)
  wig   mean lexical top-10 BM25 score divided by the query's maximum attainable
        BM25 score sum_t idf(t)(k1+1)  (a bounded weighted-information-gain, Zhou & Croft 2007)

confidence = sigmoid(bias + sum_i w_i * s_i) over the signals that are available.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from contextweave.core.types import Query, ScoredPassage
from contextweave.controller.fusion import SEMANTIC, lexical_key


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ConfidenceEstimator:
    def __init__(self, weights: dict[str, float], index, scorer, use_qpp: bool = False,
                 qpp_weights: Optional[dict[str, float]] = None):
        self.w = dict(weights)
        self.index, self.scorer, self.use_qpp = index, scorer, use_qpp
        self.qpp_w = qpp_weights or {"nqc": 1.5, "wig": 1.0}

    # ------------------------------------------------------------------ signals
    def signals(self, q: Query, current: list[ScoredPassage], lists: dict[str, list[ScoredPassage]]) -> dict[str, float]:
        top = current[:10]
        if not top:
            return {"margin": 0.0, "coverage": 0.0, "concentration": 0.0}
        norm = [r.norm_score if r.norm_score is not None else r.score for r in top]
        s10 = norm[-1] if len(norm) >= 10 else 0.0
        sig = {"margin": float(max(0.0, min(1.0, norm[0] - s10))),
               "coverage": self._coverage(q, top[:5]),
               "concentration": Counter(r.doc_id for r in top).most_common(1)[0][1] / len(top)}
        lk = lexical_key(lists)
        if lk and SEMANTIC in lists:
            a = {r.passage_id for r in lists[lk][:10]}
            b = {r.passage_id for r in lists[SEMANTIC][:10]}
            sig["agreement"] = len(a & b) / 10.0
        if self.use_qpp and lk:
            sig.update(self._qpp(q, lists[lk]))
        return sig

    def _coverage(self, q: Query, top: list[ScoredPassage]) -> float:
        terms = [t for t in dict.fromkeys(q.terms) if self.index.df(t) > 0]
        if not terms:
            return 0.0
        idf = {t: max(self.index.idf(t), 1e-3) for t in terms}
        total = sum(idf.values())
        pos = self.index.pos
        vals = []
        for r in top:
            i = pos.get(r.passage_id)
            if i is None:
                continue
            got = 0.0
            for t in terms:
                p = self.index.get(t)
                j = int(np.searchsorted(p.docs, i))
                if j < p.df and int(p.docs[j]) == i:
                    got += idf[t]
            vals.append(got / total)
        return float(np.mean(vals)) if vals else 0.0

    def _qpp(self, q: Query, lex: list[ScoredPassage]) -> dict[str, float]:
        raw = np.array([r.score for r in lex[:10]], dtype=float)
        if raw.size == 0 or raw.mean() <= 0:
            return {"nqc": 0.0, "wig": 0.0}
        nqc = float(raw.std() / raw.mean())
        terms = [t for t in dict.fromkeys(q.terms) if self.index.df(t) > 0]
        k1 = self.scorer.k1
        max_score = sum(self.scorer.idf(t) * (k1 + 1) for t in terms) or 1.0
        return {"nqc": min(nqc, 1.0), "wig": float(min(raw.mean() / max_score, 1.0))}

    # --------------------------------------------------------------- confidence
    def combine(self, sig: dict[str, float]) -> float:
        z = self.w.get("bias", 0.0)
        for k, v in sig.items():
            if k in self.w:
                z += self.w[k] * v
            elif self.use_qpp and k in self.qpp_w:
                z += self.qpp_w[k] * v
        return _sigmoid(z)

    def __call__(self, q: Query, current, lists) -> tuple[float, dict[str, float]]:
        sig = self.signals(q, current, lists)
        return self.combine(sig), sig
