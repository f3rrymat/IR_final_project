"""Tune thresholds and fit the novel components — on VALIDATION queries only.

For one collection this script
  1. measures the fixed hybrid (S5) on the validation queries (the quality target);
  2. grid-searches (tau, budget) for every adaptive system: pick the cheapest
     setting whose mean NDCG@10 is within ``tolerance`` of S5 (else the best one);
  3. fits the QPP reference distributions for percentile-calibrated fusion (N4);
  4. fits the Value-of-Computation gain table by exhaustively simulating every
     escalation order on every validation query (N2) and tunes its lambda;
  5. selects a risk-controlled tau with Learn-then-Test (N3).
Everything is written to experiments/results/models/<collection>/ and a
timestamped summary to experiments/results/.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextweave.core.config import load_config  # noqa: E402
from contextweave.controller import risk as risk_mod  # noqa: E402
from contextweave.controller.engine import models_dir  # noqa: E402
from contextweave.controller.qpp_fusion import QppFusion, relative_gap  # noqa: E402
from contextweave.controller.voc import GainModel  # noqa: E402
from contextweave.evaluation.benchmark import load_qrels, load_queries  # noqa: E402
from contextweave.evaluation.metrics import ndcg_at  # noqa: E402
from contextweave.evaluation.runner import config_snapshot, enable_cache, run_system, save_results  # noqa: E402
from contextweave.system import load_system  # noqa: E402
from experiments.systems import ADAPTIVE, SYSTEMS, budget_levels  # noqa: E402

log = logging.getLogger("fit_and_tune")
TOLERANCE = 0.01  # allowed mean NDCG@10 drop vs fixed hybrid when tuning


def ndcg_of(rows: dict) -> float:
    vals = [r["NDCG@10"] for r in rows.values() if "NDCG@10" in r]
    return float(np.mean(vals)) if vals else 0.0


def pick(grid: list[dict], target: float) -> dict:
    ok = [g for g in grid if g["ndcg"] >= target - TOLERANCE]
    return min(ok, key=lambda g: (g["cost"], -g["ndcg"])) if ok else max(grid, key=lambda g: (g["ndcg"], -g["cost"]))


def judged(queries, qrels):
    return [q for q in queries if q.qid in qrels and any(v >= 1 for v in qrels[q.qid].values())]


def fit_qpp(system, queries) -> QppFusion:
    eng = system.engine()
    ref = {"bm25": [], "dense": [], "expansion": [], "bm25_champion": []}
    for q in queries:
        query = system.query(q.text)
        for m in ref:
            cur = None
            if m == "expansion":
                cur, _, _ = eng.fuse(query, {"bm25": eng.run_method("bm25", query)[0]})
            ref[m].append(relative_gap(eng.run_method(m, query, cur)[0]))
    return QppFusion(ref, system.cfg["novelty"]["qpp_fusion"]["mix"])


def simulate_voc(system, queries, qrels, alpha=None) -> list[dict]:
    """Every state reachable by every escalation order; record the NDCG gain of each action."""
    eng = system.engine(expansion=True, alpha=alpha)
    obs = []
    for q in queries:
        rels = qrels[q.qid]
        query = system.query(q.text)
        first, _ = eng.router.first_method(query)
        ladder = eng.router.ladder(first)
        path = "semantic" if first == "dense" else "lexical"
        base = {first: eng.run_method(first, query)[0]}
        memo = {}

        def state(order: tuple[str, ...]):
            if order in memo:
                return memo[order]
            if not order:
                lists = dict(base)
            else:
                prev_lists, prev_cur, _ = state(order[:-1])
                lists = dict(prev_lists)
                lists[order[-1]] = eng.run_method(order[-1], query, prev_cur)[0]
            cur, _, _ = eng.fuse(query, lists)
            conf, _ = eng.conf(query, cur, lists)
            memo[order] = (lists, cur, (conf, ndcg_at([r.passage_id for r in cur], rels, 10)))
            return memo[order]

        for n in range(len(ladder)):
            for order in itertools.permutations(ladder, n):
                _, _, (conf, nd) = state(order)
                for a in ladder:
                    if a in order:
                        continue
                    _, _, (_, nd2) = state(order + (a,))
                    obs.append({"qid": q.qid, "action": a, "path": path, "conf": conf, "gain": nd2 - nd,
                                "depth": len(order)})
    return obs


def main(collection: str) -> dict:
    cfg = load_config(overrides={"collection": collection})
    system = load_system(cfg, collection=collection)
    enable_cache()
    qrels = load_qrels(collection)
    val = judged(load_queries(collection, "val"), qrels)
    if not val:
        print(f"[{collection}] no judged validation queries — label benchmark/qrels.tsv first; skipping.")
        return {}
    out_dir = models_dir(collection)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = {**cfg["costs"], **(cfg.get("costs_by_collection") or {}).get(collection, {})}
    budgets = budget_levels(costs)
    taus = cfg["novelty"]["risk"]["tau_grid"]
    summary: dict = {"collection": collection, "n_val": len(val), "costs": costs, "budgets": budgets}

    # 1. quality target: fixed hybrid on validation
    s5 = run_system(system, SYSTEMS["S5"], val, qrels)
    target = ndcg_of(s5["per_query"])
    summary["S5_val_ndcg"] = target

    # 3. QPP references first (N4 needs them during tuning)
    qpp = fit_qpp(system, val)
    qpp.save(out_dir / "qpp_fusion.json")

    # 2a. alpha(q) rule values: best full-escalation NDCG@10 on validation (ties -> spec defaults)
    alpha_grid = []
    defaults = cfg["controller"]["alpha"]
    for ex, de, se in itertools.product([0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6]):
        a = {"exact": ex, "default": de, "semantic": se}
        r = run_system(system, {"mode": "auto", "tau": 1.01, "budget": 1e9,
                                "controller": {"expansion": False, "alpha": a}}, val, qrels)
        dist = sum(abs(a[k] - defaults[k]) for k in a)
        alpha_grid.append({**a, "ndcg": ndcg_of(r["per_query"]), "dist": dist})
    best_a = max(alpha_grid, key=lambda g: (round(g["ndcg"], 4), -g["dist"]))
    alpha = {k: best_a[k] for k in ("exact", "default", "semantic")}
    summary["alpha"] = {"chosen": alpha, "grid": alpha_grid}
    log.info("alpha tuned: %s (full-escalation val NDCG %.4f)", alpha, best_a["ndcg"])

    def with_alpha(spec: dict) -> dict:
        return {**spec, "controller": {**spec.get("controller", {}), "alpha": alpha}}

    # 2b. tune (tau, budget) for each adaptive system with the standard ladder
    tuned = {"alpha": alpha}
    for sid in ADAPTIVE:
        spec = with_alpha(SYSTEMS[sid])
        if spec["controller"].get("features", {}).get("voc_policy") or spec.get("risk"):
            continue
        grid = []
        for b in budgets:
            for t in (taus if spec["mode"] == "auto" else [1.0]):
                r = run_system(system, {**spec, "tau": t, "budget": b}, val, qrels)
                grid.append({"tau": t, "budget": b, "ndcg": ndcg_of(r["per_query"]), "cost": r["mean_cost"]})
        best = pick(grid, target)
        tuned[sid] = {"tau": best["tau"], "budget": best["budget"], "val_ndcg": best["ndcg"],
                      "val_cost": best["cost"], "grid": grid}
        log.info("%s tuned: %s", sid, {k: v for k, v in best.items()})

    # 4. Value-of-computation gain table + lambda
    obs = simulate_voc(system, val, qrels, alpha)
    nov = cfg["novelty"]["voc"]
    gm = GainModel(nov["confidence_bins"], nov["prior_strength"]).fit(obs)
    gm.save(out_dir / nov["model_file"])
    bmax = max(budgets)
    lam_grid = []
    for lam in nov["lambda_grid"]:
        spec = {**SYSTEMS["N2"], "budget": bmax,
                "controller": {**SYSTEMS["N2"]["controller"], "voc_lambda": lam, "alpha": alpha}}
        r = run_system(system, spec, val, qrels)
        lam_grid.append({"lambda": lam, "ndcg": ndcg_of(r["per_query"]), "cost": r["mean_cost"]})
    best = min([g for g in lam_grid if g["ndcg"] >= target - TOLERANCE] or [max(lam_grid, key=lambda g: g["ndcg"])],
               key=lambda g: g["cost"])
    tuned["N2"] = {"lambda": best["lambda"], "budget": bmax, "val_ndcg": best["ndcg"], "val_cost": best["cost"],
                   "grid": lam_grid, "n_observations": len(obs)}

    # 5. Learn-then-Test risk-controlled tau (budget non-binding; tau alone controls escalation)
    rk = cfg["novelty"]["risk"]
    s5_pq = {q: r["NDCG@10"] for q, r in s5["per_query"].items() if "NDCG@10" in r}
    losses, cost_by_tau = {}, {}
    for t in taus:
        r = run_system(system, {**with_alpha(SYSTEMS["N3"]), "tau": t, "budget": bmax}, val, qrels)
        qids = [q for q in s5_pq if "NDCG@10" in r["per_query"][q]]
        losses[t] = risk_mod.failure_indicators([r["per_query"][q]["NDCG@10"] for q in qids],
                                                [s5_pq[q] for q in qids], rk["epsilon"])
        cost_by_tau[t] = r["mean_cost"]
    ltt = risk_mod.ltt_select(losses, cost_by_tau, rk["delta"], rk["eta"], sorted(taus, reverse=True))
    ltt.update({"epsilon": rk["epsilon"], "budget": bmax})
    ltt["curve"] = {str(d): risk_mod.ltt_select(losses, cost_by_tau, d, rk["eta"], sorted(taus, reverse=True))["tau"]
                    for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)}
    ltt["curve_certified"] = {str(d): risk_mod.ltt_select(losses, cost_by_tau, d, rk["eta"], sorted(taus, reverse=True))["certified"]
                              for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)}
    risk_mod.save(ltt, out_dir / rk["model_file"])
    tuned["N3"] = {"tau": ltt["tau"], "budget": bmax, "certified": ltt["certified"]}

    (out_dir / "tuned.json").write_text(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "grid"}
                                                        if k != "alpha" else v) for k, v in tuned.items()}, indent=1), encoding="utf-8", newline="")
    summary.update({"tuned": tuned, "ltt": ltt, "voc_table": gm.table, "config": config_snapshot(cfg)})
    save_results(summary, f"tuning_{collection}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for c in (sys.argv[1:] or ["cs"]):
        s = main(c)
        if s:
            print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "grid"} if k != "alpha" else v)
                              for k, v in s["tuned"].items()}, indent=1))
