"""Rule-based router: picks the first method and the escalation ladder (spec §9)."""
from __future__ import annotations

from contextweave.core.types import Query


class Router:
    def __init__(self, cfg_controller: dict, expansion_enabled: bool = True, use_routing: bool = True,
                 use_pruning: bool = True, alpha: dict | None = None):
        self.c = {**cfg_controller, "alpha": {**cfg_controller["alpha"], **(alpha or {})}}
        self.r = cfg_controller["router"]
        self.expansion_enabled, self.use_routing, self.use_pruning = expansion_enabled, use_routing, use_pruning

    def query_class(self, q: Query) -> str:
        """'exact' | 'semantic' | 'default' — also used for alpha(q)."""
        f = q.features
        if f.get("quotes") or f.get("boolean") or f.get("formula"):
            return "exact"
        if f.get("question") or f.get("n_terms", 0) >= self.r["long_min_terms"] or \
                f.get("unknown_share", 0) >= self.r["unknown_share"]:
            return "semantic"
        if f.get("n_terms", 0) <= self.r["short_max_terms"] and f.get("mean_idf", 0) >= self.r["high_idf"]:
            return "exact_short"
        return "default"

    def first_method(self, q: Query) -> tuple[str, str]:
        """Return (method, reason)."""
        if not self.use_routing:
            return self._lex("bm25_champion"), "routing disabled: always start lexical"
        cls = self.query_class(q)
        if cls == "exact":
            return "bm25", "quoted phrase / Boolean operators / formula-like tokens"
        if cls == "semantic":
            return "dense", "question form, long query or many unknown terms"
        if cls == "exact_short":
            return self._lex("bm25_champion"), "short, high-IDF, no question word"
        return self._lex("bm25_champion"), "default: cheapest lexical method"

    def _lex(self, m: str) -> str:
        return m if self.use_pruning or m != "bm25_champion" else "bm25"

    def ladder(self, first: str) -> list[str]:
        """Remaining steps after ``first``, cheapest first, as configured."""
        base = self.c["ladder_semantic"] if first == "dense" else self.c["ladder_lexical"]
        steps = [self._lex(s) for s in base]
        steps = list(dict.fromkeys(steps))  # de-duplicate (e.g. pruning disabled)
        if first in steps:
            steps = steps[steps.index(first) + 1:]
        if not self.expansion_enabled:
            steps = [s for s in steps if s != "expansion"]
        return [s for s in steps if s != first]

    def alpha_rule(self, q: Query) -> float:
        """Lexical weight alpha(q) from simple rules on the query features."""
        a = self.c["alpha"]
        cls = self.query_class(q)
        if cls in ("exact", "exact_short"):
            return float(a["exact"])
        if cls == "semantic":
            return float(a["semantic"])
        return float(a["default"])
