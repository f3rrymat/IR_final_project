"""Paired significance tests on per-query scores (scipy)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_tests(a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 42) -> dict:
    """Paired t-test, Wilcoxon signed-rank and a bootstrap 95% CI of mean(a - b)."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    d = a_ - b_
    out = {"n": int(d.size), "mean_a": float(a_.mean()), "mean_b": float(b_.mean()), "mean_diff": float(d.mean())}
    if d.size < 2 or np.allclose(d, 0):
        out.update({"t_p": 1.0, "wilcoxon_p": 1.0, "ci95": [0.0, 0.0]})
        return out
    out["t_p"] = float(stats.ttest_rel(a_, b_).pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a_, b_, zero_method="wilcox").pvalue)
    except ValueError:
        out["wilcoxon_p"] = 1.0
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda x: x[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
