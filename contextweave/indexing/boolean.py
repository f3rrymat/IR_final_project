"""Boolean retrieval: AND / OR / NOT with parentheses over postings lists.

Grammar (NOT binds tightest, then AND, then OR; adjacent terms mean AND)::

    expr   := term_and ("OR" term_and)*
    term_and := factor (["AND"] factor)*
    factor := "NOT" factor | "(" expr ")" | WORD | "quoted phrase"

Intersection, union and difference are hand-written linear merges on sorted
postings (the textbook algorithms). A quoted phrase is treated as the AND of
its terms (positional phrase queries are a planned extension).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from contextweave.core.retriever import finalize
from contextweave.core.types import Query, ScoredPassage
from contextweave.indexing.inverted_index import InvertedIndex

_TOK = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')
OPS = {"AND", "OR", "NOT"}


def intersect(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer intersection of sorted lists."""
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1  # noqa: E702
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: list[int], b: list[int]) -> list[int]:
    """Linear merge union of sorted lists."""
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1  # noqa: E702
        elif a[i] < b[j]:
            out.append(a[i]); i += 1  # noqa: E702
        else:
            out.append(b[j]); j += 1  # noqa: E702
    out.extend(a[i:]); out.extend(b[j:])  # noqa: E702
    return out


def difference(a: list[int], b: list[int]) -> list[int]:
    """a AND NOT b, linear merge."""
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1  # noqa: E702
        elif a[i] == b[j]:
            i += 1; j += 1  # noqa: E702
        else:
            j += 1
    return out


def has_boolean_syntax(raw: str) -> bool:
    return bool(re.search(r"\b(AND|OR|NOT)\b", raw))


class BooleanParser:
    """Recursive-descent parser that evaluates directly to sorted id lists."""

    def __init__(self, index: InvertedIndex, normalize: Callable[[str], list[str]]):
        self.index = index
        self.normalize = normalize
        self.all_ids = list(range(index.N))

    def evaluate(self, raw: str) -> tuple[list[int], list[str]]:
        """Return (matching passage indices, positive query terms)."""
        self.toks = _TOK.findall(raw)
        self.pos = 0
        self.terms: list[str] = []
        res = self._expr() if self.toks else []
        return res, self.terms

    def _peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _expr(self) -> list[int]:
        res = self._and()
        while self._peek() == "OR":
            self.pos += 1
            res = union(res, self._and())
        return res

    def _and(self) -> list[int]:
        res = self._factor()
        while (t := self._peek()) is not None and t not in {"OR", ")"}:
            if t == "AND":
                self.pos += 1
            if self._peek() == "NOT":
                self.pos += 1
                res = difference(res, self._factor(negated=True))
            else:
                res = intersect(res, self._factor())
        return res

    def _factor(self, negated: bool = False) -> list[int]:
        t = self._peek()
        if t is None:
            return []
        self.pos += 1
        if t == "NOT":
            return difference(self.all_ids, self._factor(negated=True))
        if t == "(":
            res = self._expr()
            if self._peek() == ")":
                self.pos += 1
            return res
        if t == ")":
            return []
        terms = self.normalize(t.strip('"'))
        if not terms:  # stop word only -> neutral element
            return self.all_ids
        if not negated:
            self.terms.extend(terms)
        res: Optional[list[int]] = None
        for term in terms:
            p = self.index.get(term)
            ids = p.docs.tolist() if p is not None else []
            res = ids if res is None else intersect(res, ids)
        return res or []


class BooleanRetriever:
    """Registered as ``boolean``. Pure Boolean: matches are unranked, so results are
    ordered by coordination level (distinct query terms present) then corpus order.
    Plain queries without operators are interpreted as the AND of their terms."""

    name = "boolean"

    def __init__(self, index: InvertedIndex, normalize: Callable[[str], list[str]], cost: float = 0.2):
        self.index = index
        self.normalize = normalize
        self.cost = cost
        self.parser = BooleanParser(index, normalize)
        self.last_stats: dict = {}

    def search(self, query: Query, k: int, candidate_ids: Optional[set[str]] = None) -> list[ScoredPassage]:
        raw = query.raw if has_boolean_syntax(query.raw) else " AND ".join(query.terms)
        ids, pos_terms = self.parser.evaluate(raw)
        allowed = None
        if candidate_ids is not None:
            allowed = {i for i, pid in enumerate(self.index.passage_ids) if pid in candidate_ids}
        uniq = sorted(set(pos_terms))
        results = []
        for i in ids:
            if allowed is not None and i not in allowed:
                continue
            coord = sum(1 for t in uniq if (p := self.index.get(t)) is not None and _contains(p.docs, i))
            # tiny tie-break keeps corpus order stable
            results.append(ScoredPassage(self.index.passage_ids[i], self.index.doc_ids[i], float(coord) - i * 1e-9, self.name,
                                         explanation={"matched_terms": coord, "boolean_query": raw}))
        self.last_stats = {"matches": len(ids), "query": raw}
        return finalize(results, k)


def _contains(sorted_arr, x: int) -> bool:
    import numpy as np

    j = int(np.searchsorted(sorted_arr, x))
    return j < len(sorted_arr) and int(sorted_arr[j]) == x
