"""The Retriever interface every retrieval method implements (spec section 6)."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from contextweave.core.types import Query, ScoredPassage


@runtime_checkable
class Retriever(Protocol):
    """A named, costed retrieval method.

    ``candidate_ids`` restricts scoring to a given set of passages (used for
    re-scoring and by the controller). Implementations must return at most
    ``k`` results sorted by descending score with ``rank`` set from 1.
    """

    name: str  # registry key, e.g. "bm25", "bm25_champion", "dense"
    cost: float  # cost units per call (section 9)

    def search(
        self, query: Query, k: int, candidate_ids: Optional[set[str]] = None
    ) -> list[ScoredPassage]: ...


def finalize(results: list[ScoredPassage], k: int) -> list[ScoredPassage]:
    """Sort by score (ties broken by passage id for determinism), keep top-k, set ranks."""
    results.sort(key=lambda r: (-r.score, r.passage_id))
    out = results[:k]
    for i, r in enumerate(out, start=1):
        r.rank = i
    return out
