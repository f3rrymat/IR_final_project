"""Dense retrieval with FAISS. Registered as ``dense``."""
from __future__ import annotations

from typing import Optional

from contextweave.core.retriever import finalize
from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage


class DenseRetriever:
    name = "dense"

    def __init__(self, vs, embedder, cost: float = 2.0):
        self.vs, self.embedder, self.cost = vs, embedder, cost
        self.last_stats: dict = {}

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        with timed() as t:
            cached = "qvec" in query.cache
            with timed() as te:
                qv = self.embedder.encode_query(query)
            scores, idx = self.vs.search(qv, k, candidate_ids)
            res = [ScoredPassage(self.vs.passage_ids[i], self.vs.doc_ids[i], float(s), self.name,
                                 explanation={"cosine": round(float(s), 4)}) for s, i in zip(scores, idx)]
            res = finalize(res, k)
        self.last_stats = {"encode_ms": te.ms, "query_vector_cached": cached, "latency_ms": t.ms,
                           "passages_scored": self.vs.N if candidate_ids is None else len(candidate_ids),
                           "N": self.vs.N}
        return res
