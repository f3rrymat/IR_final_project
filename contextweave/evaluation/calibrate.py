"""Cost calibration (spec §9): 1 cost unit = median time of a full BM25 query on the
validation queries; each method's cost = its median time / that unit.

Measured warm (several repetitions per query, first repetition discarded).
``expansion`` is measured as it runs inside the ladder: with the query vector
already computed and the current result list given as feedback.
"""
from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path

from contextweave.core import registry
from contextweave.core.config import DEFAULT_PATH, Config
from contextweave.core.timing import timed
from contextweave.evaluation.benchmark import load_queries

log = logging.getLogger(__name__)
METHODS = ["boolean", "tfidf", "bm25", "bm25_ie", "bm25_champion", "dense", "expansion"]


def measure(system, queries: list[str], repeats: int = 5, depth: int = 100) -> dict[str, float]:
    times: dict[str, list[float]] = {m: [] for m in METHODS if m in registry.available()}
    for text in queries:
        for m in times:
            r = registry.get(m)
            per = []
            for i in range(repeats + 1):
                q = system.query(text)  # fresh Query: no cached query vector
                if m == "expansion":
                    system.embedder.encode_query(q)
                    q.cache["feedback"] = registry.get("bm25").search(q, depth)
                with timed() as t:
                    r.search(q, depth)
                if i > 0:
                    per.append(t.ms)
            times[m].append(statistics.median(per))
    return {m: statistics.median(v) for m, v in times.items()}


def calibrate_costs(cfg: Config, collection: str | None = None, write: bool = True, repeats: int = 5) -> dict:
    from contextweave.system import load_system

    collection = collection or cfg["collection"]
    system = load_system(cfg, collection=collection)
    qs = [q.text for q in load_queries(collection, "val")][:100]
    med = measure(system, qs, repeats)
    unit = med["bm25"]
    costs = {m: round(t / unit, 3) for m, t in med.items()}
    out = {"collection": collection, "unit_ms": round(unit, 4), "median_ms": {m: round(t, 4) for m, t in med.items()},
           "costs": costs, "n_queries": len(qs), "repeats": repeats}
    if write:
        _write_costs(collection, costs, cfg)
    return out


def _write_costs(collection: str, costs: dict[str, float], cfg: Config, path: Path = DEFAULT_PATH) -> None:
    """Update ``costs_by_collection.<collection>`` (and ``costs`` for the default collection)."""
    text = path.read_text(encoding="utf-8")
    flow = "{" + ", ".join(f"{k}: {v}" for k, v in costs.items()) + "}"
    if collection == cfg["collection"]:
        text = re.sub(r"^costs: \{[^}]*\}", f"costs: {flow}", text, flags=re.M)
    if "costs_by_collection:" not in text:
        text = text.replace("\n# ----", "\ncosts_by_collection:   # measured by `make calibrate-costs`\n\n# ----", 1)
    pat = re.compile(rf"^  {collection}: \{{[^}}]*\}}\n", re.M)
    line = f"  {collection}: {flow}\n"
    if pat.search(text):
        text = pat.sub(line, text)
    else:
        text = text.replace("costs_by_collection:   # measured by `make calibrate-costs`\n",
                            "costs_by_collection:   # measured by `make calibrate-costs`\n" + line, 1)
    path.write_text(text, encoding="utf-8", newline="")
    log.info("wrote costs for %s: %s", collection, flow)
