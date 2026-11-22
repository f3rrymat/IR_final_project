"""System definitions S0–S8 (spec §11) and the novel ContextWeave+ systems N1–N5."""
from __future__ import annotations

import itertools

SYSTEMS: dict[str, dict] = {
    "S0": {"mode": "boolean", "label": "Boolean AND"},
    "S1": {"mode": "tfidf", "label": "TF-IDF"},
    "S2": {"mode": "bm25", "label": "BM25"},
    "S3": {"mode": "bm25_champion", "label": "BM25 + Champion + IndexElim"},
    "S4": {"mode": "dense", "label": "Dense only"},
    "S5": {"mode": "hybrid", "label": "Fixed hybrid"},
    "S6": {"mode": "route", "label": "Routing only", "controller": {"expansion": False}},
    "S7": {"mode": "auto", "label": "Full controller", "controller": {"expansion": False}},
    "S8": {"mode": "auto", "label": "S7 + expansion", "controller": {"expansion": True}},
    # --- novel (ContextWeave+) -------------------------------------------------
    "N1": {"mode": "auto", "label": "S8 + QPP confidence signals",
           "controller": {"expansion": True, "features": {"qpp_signals": True}}},
    "N2": {"mode": "auto", "label": "S8 + value-of-computation policy",
           "controller": {"expansion": True, "features": {"voc_policy": True}}},
    "N3": {"mode": "auto", "label": "S8 + risk-controlled tau (LTT)", "risk": True,
           "controller": {"expansion": True, "features": {"risk_controlled_tau": True}}},
    "N4": {"mode": "auto", "label": "S8 + QPP-percentile fusion",
           "controller": {"expansion": True, "features": {"qpp_fusion": True}}},
}
ADAPTIVE = ["S6", "S7", "S8", "N1", "N2", "N3", "N4"]

ABLATIONS = {  # remove one part at a time from S8 (same tuned tau / budget as S8)
    "S8 - expansion": {"expansion": False},
    "S8 - routing": {"expansion": True, "routing": False},
    "S8 - escalation": {"expansion": True, "escalation": False},
    "S8 - pruning": {"expansion": True, "pruning": False},
}


def budget_levels(costs: dict[str, float]) -> list[float]:
    """Every total cost a query can reach with the ladder methods (rounded)."""
    items = ["bm25_champion", "bm25", "dense", "expansion"]
    sums = set()
    for n in range(1, len(items) + 1):
        for combo in itertools.combinations(items, n):
            sums.add(round(sum(costs[m] for m in combo), 3))
    return sorted(sums)
