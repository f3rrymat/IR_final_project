"""Run every system (S0–S8, N1–N4) on a split, plus latency, significance, ablations,
quality-vs-cost sweeps and plots. All numbers come from here and are saved to
experiments/results/ with a timestamp, the hardware and the config used.

    python experiments/run_all.py cs scifact nfcorpus --split test
    python experiments/run_all.py scifact --split val --quick      # eval-quick
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextweave.core.config import load_config  # noqa: E402
from contextweave.controller.engine import models_dir  # noqa: E402
from contextweave.evaluation import plots  # noqa: E402
from contextweave.evaluation.benchmark import load_qrels, load_queries  # noqa: E402
from contextweave.evaluation.runner import (RESULTS, compare, config_snapshot, disable_cache,  # noqa: E402
                                            enable_cache, run_system, save_results)
from contextweave.system import load_system  # noqa: E402
from experiments.systems import ABLATIONS, ADAPTIVE, SYSTEMS, budget_levels  # noqa: E402

log = logging.getLogger("run_all")
QUICK = ["S2", "S4", "S5", "S7", "S8"]


def with_tuning(sid: str, tuned: dict) -> dict:
    spec = json.loads(json.dumps(SYSTEMS[sid]))
    t = tuned.get(sid)
    if t:
        for k in ("tau", "budget"):
            if k in t:
                spec[k] = t[k]
        if "lambda" in t:
            spec.setdefault("controller", {})["voc_lambda"] = t["lambda"]
    if sid in ADAPTIVE and tuned.get("alpha"):
        spec.setdefault("controller", {})["alpha"] = tuned["alpha"]
    return spec


def label_types(system, queries) -> None:
    """BEIR queries have no human type: use the router's query class instead."""
    eng = system.engine()
    for q in queries:
        if q.type in ("", "unknown"):
            c = eng.router.query_class(system.query(q.text))
            q.type = {"exact_short": "exact"}.get(c, c)


def run_systems(cfg, collection: str, split: str = "test", systems=None, latency: bool = True,
                ablations: bool = True, sweep: bool = True) -> Path:
    cfg = load_config(overrides={"collection": collection})
    system = load_system(cfg, collection=collection)
    qrels = load_qrels(collection)
    queries = load_queries(collection, split)
    label_types(system, queries)
    tuned_path = models_dir(collection) / "tuned.json"
    tuned = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else {}
    if not tuned:
        log.warning("no tuned.json for %s — using config tau/budget (run fit_and_tune.py first)", collection)
    sids = systems or list(SYSTEMS)
    if any(s in sids for s in ("N2", "N3")) and not (models_dir(collection) / "risk_tau.json").exists():
        sids = [s for s in sids if s not in ("N2", "N3")]

    # quality (memoised retrievers: exact, deterministic)
    enable_cache()
    results = {sid: run_system(system, with_tuning(sid, tuned), queries, qrels) for sid in sids}
    extra: dict = {}
    if ablations and "S8" in results:
        base = with_tuning("S8", tuned)
        extra["ablations"] = {}
        for name, ctrl in ABLATIONS.items():
            spec = {**base, "controller": {**base.get("controller", {}), **ctrl}, "label": name}
            extra["ablations"][name] = _strip(run_system(system, spec, queries, qrels))
    if sweep and "S8" in results:
        costs = {**cfg["costs"], **(cfg.get("costs_by_collection") or {}).get(collection, {})}
        pts = []
        for b in budget_levels(costs):
            for t in cfg["novelty"]["risk"]["tau_grid"]:
                r = run_system(system, {**with_tuning("S8", tuned), "tau": t, "budget": b}, queries, qrels)
                pts.append({"tau": t, "budget": b, "cost": r["mean_cost"],
                            "ndcg": r["aggregate"].get("NDCG@10", 0.0)})
        extra["sweep_S8"] = pts
    risk_file = models_dir(collection) / cfg["novelty"]["risk"]["model_file"]
    if "S5" in results and risk_file.exists() and any(r["aggregate"] for r in results.values()):
        ltt = json.loads(risk_file.read_text(encoding="utf-8"))
        eps = ltt["epsilon"]
        s5 = {q: r["NDCG@10"] for q, r in results["S5"]["per_query"].items() if "NDCG@10" in r}
        curve = []
        for d, tau in ltt.get("curve", {}).items():
            r = run_system(system, {**with_tuning("N3", tuned), "tau": tau, "budget": ltt["budget"]}, queries, qrels)
            fails = [int(r["per_query"][q]["NDCG@10"] < s5[q] - eps) for q in s5]
            curve.append({"delta": float(d), "tau": tau, "certified_on_val": ltt["curve_certified"][d],
                          "test_failure_rate": sum(fails) / len(fails), "test_ndcg": r["aggregate"]["NDCG@10"],
                          "test_cost": r["mean_cost"]})
        extra["risk_curve"] = curve
    disable_cache()

    # latency: real retrievers, warm repetitions
    if latency:
        lat_q = queries[: min(len(queries), 100)]
        for sid in sids:
            r = run_system(system, with_tuning(sid, tuned), lat_q, qrels, measure_latency=True, repeats=3)
            results[sid]["latency_median_ms"] = r["latency_median_ms"]
            results[sid]["latency_p95_ms"] = r["latency_p95_ms"]
            results[sid]["latency_measured"] = True
            results[sid]["latency_queries"] = len(lat_q)

    has_qrels = any(r["aggregate"] for r in results.values())
    sig = {}
    if has_qrels:
        for ref in ("S5", "S2"):
            if ref in results:
                sig[ref] = compare(results, ref)
    payload = {"collection": collection, "split": split, "n_queries": len(queries), "tuned": tuned,
               "systems": results, "significance": sig, **extra, "config": config_snapshot(cfg)}
    path = save_results(payload, f"eval_{collection}_{split}")
    if has_qrels:
        make_figures(payload, collection, split)
        write_tables(payload, collection, split)
    return path


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_query"}


def make_figures(payload: dict, collection: str, split: str) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    S = payload["systems"]  # noqa: N806
    kind = {"S5": "hybrid", "S7": "ours", "S8": "ours", "N1": "ours", "N2": "ours", "N3": "ours", "N4": "ours"}
    pts = {sid: {"cost": r["mean_cost"], "ndcg": r["aggregate"]["NDCG@10"], "kind": kind.get(sid, "baseline")}
           for sid, r in S.items() if r["aggregate"] and sid != "S0"}
    if payload.get("sweep_S8"):
        plots.quality_vs_cost(payload["sweep_S8"], pts, f"{collection} ({split}): quality vs cost",
                              fig_dir / f"{collection}_{split}_quality_vs_cost.png")
    types = sorted({t for r in S.values() for t in r.get("by_type", {})})
    sel = {S[s]["label"]: {t: v["NDCG@10"] for t, v in S[s]["by_type"].items()} for s in ("S2", "S5", "S8") if s in S}
    if types and sel:
        plots.by_type_bars(sel, types, f"{collection} ({split}): NDCG@10 by query type",
                           fig_dir / f"{collection}_{split}_by_type.png")
    rows = [(f"{sid} {r['label']}", r["latency_median_ms"], r["latency_p95_ms"]) for sid, r in S.items()
            if r.get("latency_measured")]
    if rows:
        plots.latency_bars(rows, f"{collection} ({split}): latency", fig_dir / f"{collection}_{split}_latency.png")


def write_tables(payload: dict, collection: str, split: str) -> None:
    """Markdown tables for the report, generated from the saved numbers."""
    S = payload["systems"]  # noqa: N806
    sig = payload.get("significance", {}).get("S5", {})
    lines = [f"### {collection} — {split} ({next(iter(S.values()))['aggregate'].get('n_queries', 0)} judged queries)", "",
             "| ID | System | P@5 | R@10 | MAP | NDCG@10 | F1@10 | mean cost | escalated | not scored | median ms | p95 ms | Δ vs S5 (p, Holm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid, r in S.items():
        a = r["aggregate"]
        if not a:
            continue
        s = sig.get(sid)
        d = f"{s['mean_diff']:+.3f} ({s['t_p_holm']:.3f})" if s else "—"
        lines.append(f"| {sid} | {r['label']} | {a['P@5']:.3f} | {a['Recall@10']:.3f} | {a['MAP']:.3f} | "
                     f"{a['NDCG@10']:.3f} | {a['F1@10']:.3f} | {r['mean_cost']:.2f} | {r['escalation_rate']:.0%} | "
                     f"{r['share_not_scored']:.0%} | {r['latency_median_ms']:.2f} | {r['latency_p95_ms']:.2f} | {d} |")
    if payload.get("ablations"):
        lines += ["", "| Ablation (from S8) | NDCG@10 | MAP | mean cost | escalated |", "|---|---|---|---|---|"]
        for name, r in payload["ablations"].items():
            lines.append(f"| {name} | {r['aggregate']['NDCG@10']:.3f} | {r['aggregate']['MAP']:.3f} | "
                         f"{r['mean_cost']:.2f} | {r['escalation_rate']:.0%} |")
    out = RESULTS / f"tables_{collection}_{split}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("collections", nargs="*", default=["cs"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--quick", action="store_true", help="main systems only, no ablations/sweeps")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for c in args.collections:
        p = run_systems(None, c, args.split, systems=QUICK if args.quick else None,
                        ablations=not args.quick, sweep=not args.quick)
        print("saved", p)
