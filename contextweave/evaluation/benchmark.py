"""Benchmark loading: queries.jsonl + qrels.tsv for a collection, plus pooling for judging."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contextweave.core.config import ROOT


@dataclass
class BenchQuery:
    qid: str
    text: str
    type: str
    split: str
    domain: str = ""


def bench_dir(collection: str) -> Path:
    base = ROOT / "benchmark"
    return base if collection == "cs" else base / collection


def load_queries(collection: str, split: Optional[str] = None) -> list[BenchQuery]:
    out = []
    with open(bench_dir(collection) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                q = BenchQuery(d["qid"], d["text"], d.get("type", "unknown"), d.get("split", "test"), d.get("domain", ""))
                if split in (None, "all") or q.split == split:
                    out.append(q)
    return out


def load_qrels(collection: str, path: Optional[Path] = None) -> dict[str, dict[str, int]]:
    p = path or bench_dir(collection) / "qrels.tsv"
    qrels: dict[str, dict[str, int]] = {}
    if not p.exists():
        return qrels
    with open(p, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        for row in rd:
            if not row or row[0] == "qid":
                continue
            qid, pid, grade = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[pid] = grade
    return qrels


def write_qrel(collection: str, qid: str, pid: str, grade: int, annotator: str = "") -> None:
    """Append/replace one judgement (used by the judging page). Keeps the file sorted."""
    p = bench_dir(collection) / "qrels.tsv"
    rows: dict[tuple[str, str], int] = {}
    if p.exists():
        for q, d in load_qrels(collection, p).items():
            for pid_, g in d.items():
                rows[(q, pid_)] = g
    rows[(qid, pid)] = int(grade)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("qid\tpassage_id\tgrade\n")
        for (q, d), g in sorted(rows.items()):
            f.write(f"{q}\t{d}\t{g}\n")
    if annotator:  # keep a per-annotator log for Cohen's kappa
        with open(bench_dir(collection) / f"judgements_{annotator}.tsv", "a", encoding="utf-8", newline="") as f:
            f.write(f"{qid}\t{pid}\t{grade}\n")


def build_pool(engine, queries: list[BenchQuery], depth: int = 10,
               systems: tuple[str, ...] = ("bm25", "tfidf", "dense", "hybrid", "auto")) -> dict[str, list[str]]:
    """Pool the top ``depth`` passages of each system per query (spec §10)."""
    pool: dict[str, list[str]] = {}
    for q in queries:
        seen: list[str] = []
        for s in systems:
            for r in engine.search(q.text, mode=s, k=depth).results:
                if r.passage_id not in seen:
                    seen.append(r.passage_id)
        pool[q.qid] = seen
    return pool
