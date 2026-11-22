"""Evaluate retrieval systems on a benchmark split and save results with the config used.

Systems are described as dicts: {"mode": ..., "controller": {...Controller kwargs}}.
Two run kinds:
  * quality runs (``measure_latency=False``) can use memoised retrievers, which is
    exact because every retriever is deterministic for a given query and feedback;
  * latency runs always call the real retrievers, several warm repetitions per query.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import os
import platform
import statistics
from pathlib import Path
from typing import Any, Optional

import numpy as np

from contextweave.core import registry
from contextweave.core.config import ROOT
from contextweave.core.types import ScoredPassage
from contextweave.evaluation.benchmark import BenchQuery, load_qrels, load_queries
from contextweave.evaluation.metrics import aggregate, evaluate_run, per_query
from contextweave.evaluation.significance import holm, paired_tests

log = logging.getLogger(__name__)
RESULTS = ROOT / "experiments" / "results"


class CachedRetriever:
    """Memoising proxy with the same name/cost; registered in place of the original."""

    def __init__(self, inner):
        self.inner, self.name, self.cost = inner, inner.name, inner.cost
        self._memo: dict[Any, tuple[list[ScoredPassage], dict]] = {}

    @property
    def last_stats(self):
        return self._last

    def search(self, query, k, candidate_ids=None):
        fb = query.cache.get("feedback")
        key = (query.raw, tuple(sorted(query.terms)), k, None if candidate_ids is None else len(candidate_ids),
               tuple(r.passage_id for r in fb[:10]) if fb else None)
        if key not in self._memo:
            res = self.inner.search(query, k, candidate_ids)
            self._memo[key] = (res, dict(getattr(self.inner, "last_stats", {}) or {}))
        res, self._last = self._memo[key]
        return [ScoredPassage(r.passage_id, r.doc_id, r.score, r.method, r.rank, r.norm_score, r.explanation) for r in res]


def enable_cache() -> None:
    for name in registry.available():
        r = registry.get(name)
        if not isinstance(r, CachedRetriever):
            registry.register(CachedRetriever(r))


def disable_cache() -> None:
    for name in registry.available():
        r = registry.get(name)
        if isinstance(r, CachedRetriever):
            registry.register(r.inner)


def hardware() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", encoding="utf-8") as f:
            info["mem_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except OSError:
        pass
    return info


def run_system(system, spec: dict, queries: list[BenchQuery], qrels: dict, depth: int = 100,
               measure_latency: bool = False, repeats: int = 3) -> dict:
    """Run one system over the queries. Returns per-query rows + aggregates."""
    eng = system.engine(**spec.get("controller", {}))
    mode = spec["mode"]
    kw = {k: spec[k] for k in ("budget", "tau") if k in spec}
    rows: dict[str, dict] = {}
    run: dict[str, list[str]] = {}
    N = len(system.store)  # noqa: N806
    for q in queries:
        lat = []
        reps = repeats if measure_latency else 1
        for _ in range(reps):
            resp = eng.search(q.text, mode=mode, k=depth, **kw)
            lat.append(resp.total_latency_ms)
        ranked = [r.passage_id for r in resp.results]
        run[q.qid] = ranked
        methods = resp.extras["methods_run"]
        scored = [st.get("passages_scored", N) for st in resp.extras["stats"].values()] or [N]
        rows[q.qid] = {"type": q.type, "cost": resp.total_cost, "latency_ms": statistics.median(lat),
                       "methods": methods, "escalated": len(methods) > 1,
                       "share_not_scored": 1.0 - min(1.0, max(scored) / N),
                       "stop": resp.extras.get("stop_reason", "")}
        if q.qid in qrels:
            rows[q.qid].update(per_query(ranked, qrels[q.qid]))
    judged = {qid: r for qid, r in rows.items() if qid in qrels and any(g >= 1 for g in qrels[qid].values())}
    return {"label": spec.get("label", mode), "spec": spec, "per_query": rows,
            "aggregate": aggregate({k: {m: v for m, v in r.items() if m in METRICS} for k, r in judged.items()}),
            "by_type": _by_type(judged), "mean_cost": float(np.mean([r["cost"] for r in rows.values()])),
            "escalation_rate": float(np.mean([r["escalated"] for r in rows.values()])),
            "share_not_scored": float(np.mean([r["share_not_scored"] for r in rows.values()])),
            "latency_median_ms": float(np.median([r["latency_ms"] for r in rows.values()])),
            "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows.values()], 95)),
            "latency_measured": measure_latency}


METRICS = ("P@5", "P@10", "Recall@10", "Recall@100", "MAP", "NDCG@10", "MRR")


def _by_type(rows: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    types = sorted({r["type"] for r in rows.values()})
    for t in types:
        sub = {k: {m: v for m, v in r.items() if m in METRICS} for k, r in rows.items() if r["type"] == t}
        if sub:
            agg = aggregate(sub)
            agg["mean_cost"] = float(np.mean([rows[k]["cost"] for k in sub]))
            out[t] = agg
    return out


def compare(results: dict[str, dict], reference: str, metric: str = "NDCG@10") -> dict[str, dict]:
    """Paired tests of every system against ``reference`` on per-query ``metric`` (Holm-adjusted)."""
    ref = results[reference]["per_query"]
    out = {}
    for sid, r in results.items():
        if sid == reference:
            continue
        qids = [q for q in ref if metric in ref[q] and metric in r["per_query"].get(q, {})]
        if len(qids) < 2:
            continue
        out[sid] = paired_tests([r["per_query"][q][metric] for q in qids], [ref[q][metric] for q in qids])
    adj = holm({k: v["t_p"] for k, v in out.items()})
    for k in out:
        out[k]["t_p_holm"] = adj[k]
    return out


def save_results(payload: dict, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"timestamp_utc": ts, "hardware": hardware(), **payload}
    path = RESULTS / f"{name}_{ts}.json"
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8", newline="")
    (RESULTS / f"{name}_latest.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8", newline="")
    log.info("saved %s", path)
    return path


def config_snapshot(cfg) -> dict:
    snap = copy.deepcopy(dict(cfg))
    snap.pop("paths", None)
    return snap
