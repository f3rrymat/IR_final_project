"""Evaluation metrics, written by hand (spec §8.6). Cross-checked against ir-measures in tests.

Conventions (match trec_eval / ir-measures):
  * a passage is relevant if its grade >= 1 (``rel_threshold``)
  * NDCG uses gain = grade and discount 1/log2(rank + 1); the ideal ranking is
    built from all judged passages of the query
  * AP is computed over the whole returned ranking and divided by the number of
    relevant passages
  * queries without any relevant passage are skipped by the aggregators
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]  # qid -> passage_id -> grade


def precision_at(ranked: Sequence[str], rels: Mapping[str, int], k: int, thr: int = 1) -> float:
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= thr) / k


def recall_at(ranked: Sequence[str], rels: Mapping[str, int], k: int, thr: int = 1) -> float:
    n_rel = sum(1 for g in rels.values() if g >= thr)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= thr) / n_rel


def average_precision(ranked: Sequence[str], rels: Mapping[str, int], thr: int = 1) -> float:
    n_rel = sum(1 for g in rels.values() if g >= thr)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if rels.get(d, 0) >= thr:
            hits += 1
            total += hits / i
    return total / n_rel


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    ideal = dcg(sorted((g for g in rels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg(max(rels.get(d, 0), 0) for d in ranked[:k]) / ideal


def reciprocal_rank(ranked: Sequence[str], rels: Mapping[str, int], thr: int = 1) -> float:
    for i, d in enumerate(ranked, start=1):
        if rels.get(d, 0) >= thr:
            return 1.0 / i
    return 0.0


def per_query(ranked: Sequence[str], rels: Mapping[str, int]) -> dict[str, float]:
    """The metric set used throughout the project for one query."""
    return {"P@5": precision_at(ranked, rels, 5), "P@10": precision_at(ranked, rels, 10),
            "Recall@10": recall_at(ranked, rels, 10), "Recall@100": recall_at(ranked, rels, 100),
            "MAP": average_precision(ranked, rels), "NDCG@10": ndcg_at(ranked, rels, 10),
            "MRR": reciprocal_rank(ranked, rels)}


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Qrels) -> dict[str, dict[str, float]]:
    """Per-query metrics for every query that has at least one relevant passage."""
    out = {}
    for qid, rels in qrels.items():
        if not any(g >= 1 for g in rels.values()):
            continue
        out[qid] = per_query(list(run.get(qid, [])), rels)
    return out


def aggregate(per_q: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_q:
        return {}
    keys = next(iter(per_q.values())).keys()
    agg = {k: sum(v[k] for v in per_q.values()) / len(per_q) for k in keys}
    agg["F1@10"] = f1(agg["P@10"], agg["Recall@10"])  # F1 of the mean P@10 and mean Recall@10
    agg["n_queries"] = len(per_q)
    return agg


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two annotators' labels on the same items (spec §10)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and the same length")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == lab) / n) * (sum(1 for y in b if y == lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
