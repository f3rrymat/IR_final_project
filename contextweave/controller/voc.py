"""Value-of-Computation (VoC) escalation policy — novel extension (flag ``voc_policy``).

The fixed ladder always tries steps in one order and stops on a confidence
threshold. VoC treats every remaining step as an action with an *expected
quality gain* and a *cost*, and greedily takes the action with the best

    expected gain in NDCG@10 per cost unit,   E[ΔNDCG | state, a] / cost(a)

stopping when the best ratio falls below ``lambda`` or nothing is affordable.

The expected gains come from a small, explainable table fitted on validation
queries only: for each action, query path (lexical-first / semantic-first) and
confidence bin, the mean observed ΔNDCG@10. Sparse cells are shrunk toward the
(action, path) mean, which is shrunk toward the action mean (empirical-Bayes
style, ``prior_strength`` pseudo-counts). Every decision is logged with its
expected gain, so the trace explains *why* a step was (not) worth its cost.
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Optional


class GainModel:
    def __init__(self, bins: list[float], prior_strength: float = 3.0, table: Optional[dict] = None):
        self.bins, self.m = list(bins), float(prior_strength)
        self.table = table or {}

    # ---------------------------------------------------------------- fitting
    def bin_of(self, conf: float) -> int:
        return max(0, min(len(self.bins) - 2, bisect.bisect_right(self.bins, conf) - 1))

    def fit(self, observations: list[dict]) -> "GainModel":
        """observations: dicts with keys action, path, conf, gain."""
        acc: dict[str, list[float]] = defaultdict(list)
        for o in observations:
            a, p, b = o["action"], o["path"], self.bin_of(o["conf"])
            for key in (f"{a}", f"{a}|{p}", f"{a}|{p}|{b}"):
                acc[key].append(float(o["gain"]))
        self.table = {k: {"mean": fmean(v), "n": len(v)} for k, v in acc.items()}
        return self

    # ------------------------------------------------------------- prediction
    def expected_gain(self, action: str, path: str, conf: float) -> float:
        base = self.table.get(action, {"mean": 0.0, "n": 0})
        mid = self.table.get(f"{action}|{path}", {"mean": base["mean"], "n": 0})
        cell = self.table.get(f"{action}|{path}|{self.bin_of(conf)}", {"mean": 0.0, "n": 0})
        mid_mean = (mid["n"] * mid["mean"] + self.m * base["mean"]) / (mid["n"] + self.m)
        return (cell["n"] * cell["mean"] + self.m * mid_mean) / (cell["n"] + self.m)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"bins": self.bins, "prior_strength": self.m, "table": self.table}, indent=1), encoding="utf-8", newline="")

    @classmethod
    def load(cls, path: str | Path) -> "GainModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["bins"], d["prior_strength"], d["table"])


def choose_action(model: GainModel, remaining: list[str], path: str, conf: float, budget, lam: float) -> tuple[Optional[str], dict]:
    """Pick the affordable action with the best expected gain per cost (or None)."""
    scored = {}
    for a in remaining:
        g = model.expected_gain(a, path, conf)
        c = max(budget.cost(a), 1e-9)
        scored[a] = {"expected_gain": round(g, 4), "cost": c, "ratio": g / c, "affordable": budget.affordable(a)}
    options = [(v["ratio"], a) for a, v in scored.items() if v["affordable"]]
    if not options:
        return None, scored
    ratio, best = max(options)
    return (best if ratio >= lam else None), scored
