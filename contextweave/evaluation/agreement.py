"""Inter-annotator agreement (Cohen's kappa) from the per-annotator judgement logs."""
from __future__ import annotations

import csv
import itertools

from contextweave.evaluation.benchmark import bench_dir
from contextweave.evaluation.metrics import cohen_kappa


def load_annotations(collection: str = "cs") -> dict[str, dict[tuple[str, str], int]]:
    out: dict[str, dict[tuple[str, str], int]] = {}
    for p in bench_dir(collection).glob("judgements_*.tsv"):
        name = p.stem.removeprefix("judgements_")
        with open(p, encoding="utf-8") as f:
            for qid, pid, g in csv.reader(f, delimiter="\t"):
                out.setdefault(name, {})[(qid, pid)] = int(g)  # last judgement wins
    return out


def kappa_report(collection: str = "cs") -> dict:
    ann = load_annotations(collection)
    report = {}
    for a, b in itertools.combinations(sorted(ann), 2):
        common = sorted(set(ann[a]) & set(ann[b]))
        if len(common) >= 2:
            k = cohen_kappa([ann[a][x] for x in common], [ann[b][x] for x in common])
            report[f"{a} vs {b}"] = {"items": len(common), "kappa": round(k, 3),
                                     "binary_kappa": round(cohen_kappa([int(ann[a][x] > 0) for x in common],
                                                                       [int(ann[b][x] > 0) for x in common]), 3)}
    print(report or "need at least two annotators with overlapping judgements")
    return report
