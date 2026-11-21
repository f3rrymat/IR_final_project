"""The adaptive controller: route -> retrieve -> estimate confidence -> stop or escalate.

Modes
  auto     full controller (routing + confidence + budget [+ expansion if enabled])
  route    adaptive first method only, no escalation (system S6)
  hybrid   fixed hybrid: always BM25 + dense, fixed alpha (system S5)
  <name>   any registered retriever by name: bm25, tfidf, dense, boolean, bm25_champion, ...

Every decision is appended to the trace as a TraceStep.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from contextweave.core import registry
from contextweave.core.config import ROOT
from contextweave.core.timing import timed
from contextweave.core.types import Query, ScoredPassage, SearchResponse, TraceStep
from contextweave.controller.budget import Budget
from contextweave.controller.confidence import ConfidenceEstimator
from contextweave.controller.fusion import SEMANTIC, dedupe, fuse, lexical_key
from contextweave.controller.qpp_fusion import QppFusion
from contextweave.controller.router import Router
from contextweave.controller import risk as risk_mod
from contextweave.controller.voc import GainModel, choose_action

log = logging.getLogger(__name__)


def models_dir(collection: str) -> Path:
    """Where fitted novelty models (VoC gains, risk tau, QPP references) are stored."""
    return ROOT / "experiments" / "results" / "models" / collection


def tuned_defaults(collection: str, cfg) -> dict:
    """tau and budget of the full controller (S8) tuned on this collection's validation queries.

    Falls back to ``controller.tau`` / ``controller.budget`` in the config (the cs defaults) when the collection
    has not been tuned. Used by the CLI and the web API so interactive searches match the evaluated system.
    """
    import json

    p = models_dir(collection) / "tuned.json"
    t = json.loads(p.read_text(encoding="utf-8")).get("S8", {}) if p.exists() else {}
    return {"tau": float(t.get("tau", cfg["controller"]["tau"])),
            "budget": float(t.get("budget", cfg["controller"]["budget"])),
            "source": "tuned on validation (S8)" if t else "config defaults"}


class Controller:
    def __init__(self, system, tau: Optional[float] = None, budget: Optional[float] = None,
                 k: Optional[int] = None, depth: Optional[int] = None, features: Optional[dict] = None,
                 expansion: Optional[bool] = None, routing: bool = True, pruning: bool = True,
                 escalation: bool = True, costs: Optional[dict] = None, voc_model: Optional[GainModel] = None,
                 qpp_model: Optional[QppFusion] = None, voc_lambda: Optional[float] = None,
                 alpha: Optional[dict] = None):
        self.system = system
        cfg = system.cfg
        self.cfg = cfg
        c = cfg["controller"]
        self.feat = {**(cfg.get("features") or {}), **(features or {})}
        by_coll = (cfg.get("costs_by_collection") or {}).get(cfg["collection"], {})
        self.costs = {**cfg["costs"], **by_coll, **(costs or {})}
        self.k = k or cfg["retrieval"]["k"]
        self.depth = depth or cfg["retrieval"]["depth"]
        self.budget_total = float(budget if budget is not None else c["budget"])
        self.escalation = escalation
        exp_on = cfg["expansion"]["enabled"] if expansion is None else expansion
        self.router = Router(c, expansion_enabled=exp_on and system.semantic, use_routing=routing,
                             use_pruning=pruning, alpha=alpha)
        nov = cfg.get("novelty", {})
        self.conf = ConfidenceEstimator(c["confidence_weights"], system.index, system.scorer,
                                        use_qpp=bool(self.feat.get("qpp_signals")),
                                        qpp_weights={"nqc": nov.get("qpp", {}).get("nqc_weight", 1.5),
                                                     "wig": nov.get("qpp", {}).get("wig_weight", 1.0)})
        mdir = models_dir(cfg["collection"])
        self.tau = float(tau if tau is not None else c["tau"])
        self.tau_source = "config" if tau is None else "argument"
        if tau is None and self.feat.get("risk_controlled_tau"):
            f = mdir / nov["risk"]["model_file"]
            if f.exists():
                self.tau, self.tau_source = risk_mod.load_tau(f), "learn-then-test"
            else:
                log.warning("risk_controlled_tau on but %s missing; using config tau", f)
        self.voc = voc_model
        if self.voc is None and self.feat.get("voc_policy"):
            f = mdir / nov["voc"]["model_file"]
            self.voc = GainModel.load(f) if f.exists() else None
            if self.voc is None:
                log.warning("voc_policy on but %s missing; using the fixed ladder", f)
        self.voc_lambda = float(voc_lambda if voc_lambda is not None else nov.get("voc", {}).get("lambda", 0.01))
        self.qpp = qpp_model
        if self.qpp is None and self.feat.get("qpp_fusion"):
            f = mdir / "qpp_fusion.json"
            self.qpp = QppFusion.load(f) if f.exists() else None
        self._domain_cache: dict[tuple, set[str]] = {}

    # ----------------------------------------------------------- primitives
    def candidates(self, q: Query) -> Optional[set[str]]:
        doms = tuple(sorted(q.filters.get("domains") or []))
        if not doms:
            return None
        if doms not in self._domain_cache:
            self._domain_cache[doms] = self.system.store.ids_in_domains(doms)
        return self._domain_cache[doms]

    def run_method(self, name: str, q: Query, current: Optional[list[ScoredPassage]] = None) -> tuple[list[ScoredPassage], float, dict]:
        r = self.system.retrievers.get(name) or registry.get(name)
        if name == "expansion" and current:
            q.cache["feedback"] = current
        with timed() as t:
            res = r.search(q, self.depth, self.candidates(q))
        q.cache.pop("feedback", None)
        return res, t.ms, dict(getattr(r, "last_stats", {}) or {})

    def alpha(self, q: Query, lists: dict[str, list[ScoredPassage]]) -> tuple[float, dict]:
        a = self.router.alpha_rule(q)
        info: dict[str, Any] = {"alpha_rule": a}
        lk = lexical_key(lists)
        if self.qpp is not None and lk and SEMANTIC in lists:
            a, info = self.qpp.alpha(a, lk, lists[lk], lists[SEMANTIC])
        return a, info

    def fuse(self, q: Query, lists: dict[str, list[ScoredPassage]], fixed_alpha: Optional[float] = None):
        if fixed_alpha is not None:
            a, info = fixed_alpha, {"alpha_fixed": fixed_alpha}
        else:
            a, info = self.alpha(q, lists)
        cur = fuse(lists, a, self.depth, self.cfg["fusion"]["normalisation"])
        return cur, a, info

    # ------------------------------------------------------------- the loop
    def search(self, raw: str, mode: str = "auto", k: Optional[int] = None, budget: Optional[float] = None,
               tau: Optional[float] = None, filters: Optional[dict] = None, query: Optional[Query] = None) -> SearchResponse:
        k = k or self.k
        tau = self.tau if tau is None else float(tau)
        b = Budget(self.budget_total if budget is None else budget, self.costs)
        trace: list[TraceStep] = []
        lists: dict[str, list[ScoredPassage]] = {}
        stats: dict[str, dict] = {}
        with timed() as total:
            q = query or self.system.query(raw, filters)
            if mode == "hybrid":
                current = self._fixed_hybrid(q, b, trace, lists, stats)
            elif mode in ("auto", "route"):
                current = self._adaptive(q, b, trace, lists, stats, tau, escalate=(mode == "auto" and self.escalation))
            else:
                res, ms, st = self.run_method(mode, q)
                c = b.charge(mode)
                lists[mode], stats[mode] = res, st
                trace.append(TraceStep("retrieve", mode, c, b.left, None, {}, ms, f"single method: {mode}"))
                current, _, _ = self.fuse(q, {mode: res})
            results = dedupe(current)[:k]
        extras = {"lists": {m: v[:k] for m, v in lists.items()}, "stats": stats, "methods_run": list(lists),
                  "tau": tau, "tau_source": self.tau_source, "budget": b.total,
                  "stop_reason": next((s.note for s in reversed(trace) if s.action == "stop"), "")}
        return SearchResponse(q, results, trace, b.spent, total.ms, mode, extras)

    def _fixed_hybrid(self, q, b, trace, lists, stats):
        for m in ("bm25", "dense"):
            res, ms, st = self.run_method(m, q)
            c = b.charge(m)
            lists[m], stats[m] = res, st
            trace.append(TraceStep("retrieve", m, c, b.left, None, {}, ms, "fixed hybrid"))
        a = float(self.cfg["fusion"].get("hybrid_alpha", 0.5))
        current, _, _ = self.fuse(q, lists, fixed_alpha=a)
        conf, sig = self.conf(q, current, lists)
        trace.append(TraceStep("confidence", None, 0.0, b.left, conf, sig, 0.0, f"alpha fixed = {a}"))
        return current

    def _adaptive(self, q, b, trace, lists, stats, tau, escalate=True):
        first, why = self.router.first_method(q)
        ladder = self.router.ladder(first)
        cls = self.router.query_class(q)
        trace.append(TraceStep("route", first, 0.0, b.left, None, {k: round(v, 3) for k, v in q.features.items()},
                               0.0, f"class={cls}; {why}; ladder={ladder}"))
        res, ms, st = self.run_method(first, q)
        c = b.charge(first)
        lists[first], stats[first] = res, st
        trace.append(TraceStep("retrieve", first, c, b.left, None, {}, ms, f"{len(res)} results"))
        current, a, ainfo = self.fuse(q, lists)
        conf, sig = self.conf(q, current, lists)
        trace.append(TraceStep("confidence", None, 0.0, b.left, conf, sig, 0.0, f"tau={tau:.2f}"))
        if not escalate:
            trace.append(TraceStep("stop", None, 0.0, b.left, conf, {}, 0.0, "routing only (no escalation)"))
            return current
        path = "semantic" if first == "dense" else "lexical"
        if self.voc is not None:
            return self._voc_loop(q, b, trace, lists, stats, ladder, path, current, conf)
        stopped = False
        for step in ladder:
            if conf >= tau:
                trace.append(TraceStep("stop", None, 0.0, b.left, conf, {}, 0.0, f"confident ({conf:.2f} >= {tau:.2f})"))
                stopped = True
                break
            if not b.affordable(step):
                trace.append(TraceStep("escalate", step, 0.0, b.left, conf, {}, 0.0,
                                       f"skip {step}: needs {b.cost(step):.2f}, {b.left:.2f} left"))
                continue
            trace.append(TraceStep("escalate", step, 0.0, b.left, conf, {}, 0.0,
                                   f"confidence {conf:.2f} < {tau:.2f} -> run {step}"))
            current, conf = self._step(q, b, trace, lists, stats, step, current, tau)
        if not stopped:
            reason = f"confident ({conf:.2f} >= {tau:.2f})" if conf >= tau else "ladder exhausted or budget"
            trace.append(TraceStep("stop", None, 0.0, b.left, conf, {}, 0.0, reason))
        return current

    def _step(self, q, b, trace, lists, stats, step, current, tau):
        res, ms, st = self.run_method(step, q, current)
        c = b.charge(step)
        lists[step], stats[step] = res, st
        note = f"{len(res)} results"
        if step == "expansion":
            note += "; added " + ", ".join(t["word"] for t in st.get("added_terms", [])) if st.get("added_terms") else "; no terms passed the drift filter"
        trace.append(TraceStep("retrieve", step, c, b.left, None, {}, ms, note))
        current, a, ainfo = self.fuse(q, lists)
        conf, sig = self.conf(q, current, lists)
        sig = {**sig, **{f"alpha.{k}": v for k, v in ainfo.items() if isinstance(v, (int, float))}, "alpha": a}
        trace.append(TraceStep("confidence", None, 0.0, b.left, conf, sig, 0.0, f"fused {sorted(lists)} alpha={a:.2f}"))
        return current, conf

    def _voc_loop(self, q, b, trace, lists, stats, ladder, path, current, conf):
        remaining = list(ladder)
        while remaining:
            action, scored = choose_action(self.voc, remaining, path, conf, b, self.voc_lambda)
            sig = {f"E[gain]/cost {a}": round(v["ratio"], 4) for a, v in scored.items()}
            if action is None:
                any_aff = any(v["affordable"] for v in scored.values())
                reason = (f"voc: best expected gain per cost < lambda={self.voc_lambda}" if any_aff
                          else "budget: no remaining step affordable")
                trace.append(TraceStep("stop", None, 0.0, b.left, conf, sig, 0.0, reason))
                return current
            trace.append(TraceStep("escalate", action, 0.0, b.left, conf, sig, 0.0,
                                   f"voc: {action} has best expected gain/cost "
                                   f"({scored[action]['expected_gain']:+.3f} NDCG for {scored[action]['cost']:.2f})"))
            current, conf = self._step(q, b, trace, lists, stats, action, current, self.tau)
            remaining.remove(action)
        trace.append(TraceStep("stop", None, 0.0, b.left, conf, {}, 0.0, "voc: all steps used"))
        return current
