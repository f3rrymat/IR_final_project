"""Risk-controlled stopping threshold via Learn-then-Test — novel extension
(flag ``risk_controlled_tau``).

Instead of hand-picking the confidence threshold tau, choose it so that, with
probability at least 1 - eta over the draw of validation queries,

    P( NDCG@10_controller(q) < NDCG@10_fixed_hybrid(q) - epsilon ) <= delta

for new queries from the same distribution. This follows Learn-then-Test
(Angelopoulos, Bates, Candès, Jordan & Lei, 2021): each candidate tau is a
hypothesis H_tau: risk(tau) > delta; an exact binomial p-value tests it, and
fixed-sequence testing from the most conservative tau (most escalation)
toward cheaper ones controls the family-wise error rate at eta without a
multiple-testing correction. The cheapest tau whose hypothesis is rejected is
selected. If none is rejected, the most conservative tau is returned and the
result says the guarantee could not be certified.
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binom


def failure_indicators(ctrl_ndcg: list[float], ref_ndcg: list[float], epsilon: float) -> list[int]:
    return [int(c < r - epsilon) for c, r in zip(ctrl_ndcg, ref_ndcg)]


def binomial_pvalue(failures: int, n: int, delta: float) -> float:
    """p-value of H0: risk > delta given ``failures`` out of ``n`` (exact, conservative)."""
    return float(binom.cdf(failures, n, delta))


def ltt_select(losses_by_tau: dict[float, list[int]], costs_by_tau: dict[float, float], delta: float,
               eta: float, order: list[float]) -> dict:
    """Fixed-sequence Learn-then-Test over ``order`` (most conservative first)."""
    tested, chosen, certified = [], None, False
    for tau in order:
        losses = losses_by_tau[tau]
        n, f = len(losses), int(sum(losses))
        p = binomial_pvalue(f, n, delta)
        rejected = p <= eta
        tested.append({"tau": tau, "failures": f, "n": n, "empirical_risk": f / max(n, 1), "p_value": p,
                       "rejected_H0": rejected, "mean_cost": costs_by_tau.get(tau)})
        if not rejected:
            break
        chosen, certified = tau, True
    if chosen is None:
        chosen = order[0]
    return {"tau": chosen, "certified": certified, "delta": delta, "eta": eta, "tested": tested}


def save(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=1), encoding="utf-8", newline="")


def load_tau(path: str | Path) -> float:
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["tau"])
