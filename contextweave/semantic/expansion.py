"""Embedding-filtered pseudo-relevance-feedback query expansion. Registered as ``expansion``.

1. Feedback = the top ``feedback_docs`` passages of the current result list
   (the controller passes it in); without it, a BM25 first pass is used.
2. Candidate terms = the ``candidate_terms`` terms with the highest summed
   TF-IDF over the feedback passages (query terms, numbers and very rare
   terms excluded).
3. Drift guard: keep only candidates whose word embedding has cosine >=
   ``min_similarity`` with the query embedding; take the top ``max_terms`` of
   those in TF-IDF order.
4. Add them to the query with weight ``weight`` and re-run BM25.

The added terms (with surface word, TF-IDF and similarity) are returned in the
explanation so the UI can show them.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from contextweave.core.store import PassageStore
from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage
from contextweave.indexing.inverted_index import InvertedIndex
from contextweave.lexical.bm25 import BM25Scorer, to_results
from contextweave.textproc.tokenizer import tokenize


class ExpansionRetriever:
    name = "expansion"

    def __init__(self, scorer: BM25Scorer, index: InvertedIndex, store: PassageStore, embedder, vs,
                 normalize: Callable[[str], list[str]], cost: float = 1.0, max_terms: int = 5,
                 weight: float = 0.3, feedback_docs: int = 5, candidate_terms: int = 30,
                 min_similarity: float = 0.25):
        self.scorer, self.index, self.store, self.embedder, self.vs = scorer, index, store, embedder, vs
        self.normalize, self.cost = normalize, cost
        self.max_terms, self.weight, self.fb, self.n_cand, self.min_sim = (max_terms, weight, feedback_docs,
                                                                           candidate_terms, min_similarity)
        self.last_stats: dict = {}

    def candidates(self, query: Query, feedback_ids: list[str]) -> list[dict]:
        """High-TF-IDF terms of the feedback passages with their most common surface word."""
        qterms = set(query.terms)
        tfidf: dict[str, float] = defaultdict(float)
        surface: dict[str, Counter] = defaultdict(Counter)
        for pid in feedback_ids:
            words = tokenize(self.store[pid].text)
            terms_here: Counter = Counter()
            for w in words:
                for t in self.normalize(w):
                    terms_here[t] += 1
                    surface[t][w] += 1
            for t, tf in terms_here.items():
                if t in qterms or t.isdigit() or len(t) < 3 or self.index.df(t) < 2:
                    continue
                tfidf[t] += (1 + math.log(tf)) * self.index.idf(t)
        top = sorted(tfidf.items(), key=lambda x: (-x[1], x[0]))[: self.n_cand]
        return [{"term": t, "word": surface[t].most_common(1)[0][0], "tfidf": round(s, 3)} for t, s in top]

    def expand(self, query: Query, feedback_ids: list[str]) -> list[dict]:
        cands = self.candidates(query, feedback_ids)
        if not cands:
            return []
        qv = self.embedder.encode_query(query)
        sims = self.embedder.encode_terms([c["word"] for c in cands]) @ qv
        kept = []
        for c, s in zip(cands, sims):
            c["similarity"] = round(float(s), 3)
            if s >= self.min_sim and len(kept) < self.max_terms:
                kept.append(c)
        return kept

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None,
               feedback: Optional[list[ScoredPassage]] = None) -> list[ScoredPassage]:
        with timed() as t:
            if feedback is None:
                feedback = query.cache.get("feedback")  # set by the controller: current fused list
            if feedback is None:
                w0 = self.scorer.query_weights(query)
                s0, _ = self.scorer.accumulate(w0, self.index.mask(candidate_ids))
                feedback = to_results(self.index, s0, self.fb, "bm25")
            fb_ids = [r.passage_id for r in feedback[: self.fb]]
            added = self.expand(query, fb_ids)
            expanded = Query(query.raw, list(query.terms), dict(query.features), dict(query.filters),
                             {**query.term_weights, **{a["term"]: self.weight for a in added}})
            expanded.cache = query.cache
            weights = self.scorer.query_weights(expanded)
            scores, touched = self.scorer.accumulate(weights, self.index.mask(candidate_ids))
            res = to_results(self.index, scores, k, self.name)
            pos = self.index.pos
            for r in res[:10]:
                r.explanation = {"terms": self.scorer.explain(weights, pos[r.passage_id]), "added_terms": added}
        self.last_stats = {"added_terms": added, "feedback": fb_ids, "postings_touched": touched,
                           "passages_scored": int((scores > 0).sum()), "latency_ms": t.ms, "N": self.index.N}
        return res
