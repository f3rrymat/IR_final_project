"""Load BEIR-format public test collections (SciFact, NFCorpus) with human qrels.

Each BEIR document becomes one Passage (title + text). Queries and qrels are
converted to our benchmark format: ``benchmark/<name>/queries.jsonl`` and
``benchmark/<name>/qrels.tsv``. Validation = BEIR train/dev split (sampled),
test = BEIR test split. These collections give externally-judged numbers that
can be compared with published results (e.g. the BEIR paper).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from contextweave.core.types import Passage
from contextweave.ingestion.manifest import write_passages
from contextweave.textproc.tokenizer import tokenize

VAL_SPLIT = {"scifact": "train", "nfcorpus": "dev"}


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)  # header
        for qid, did, score in rd:
            out.setdefault(qid, {})[did] = int(score)
    return out


def convert(name: str, src_dir: Path, processed_dir: Path, bench_dir: Path,
            max_val: int = 300, seed: int = 42) -> dict:
    """Convert one BEIR dataset. Returns simple statistics."""
    base = src_dir / name
    passages = []
    with open(base / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + ". " + d.get("text", "")).strip(". ").strip()
            passages.append(Passage(passage_id=str(d["_id"]), doc_id=str(d["_id"]), text=text,
                                    title=d.get("title", "") or str(d["_id"]), source=name, page=None,
                                    domain=name, n_tokens=len(tokenize(text))))
    write_passages(passages, processed_dir / name / "passages.jsonl")

    qtext = {}
    with open(base / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qtext[str(d["_id"])] = d["text"]
    test = _read_qrels(base / "qrels" / "test.tsv")
    val = _read_qrels(base / "qrels" / f"{VAL_SPLIT[name]}.tsv")
    rng = random.Random(seed)
    val_ids = sorted(q for q in val if q not in test)
    if len(val_ids) > max_val:
        val_ids = sorted(rng.sample(val_ids, max_val))
    out_dir = bench_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8", newline="") as fq, open(out_dir / "qrels.tsv", "w", encoding="utf-8", newline="") as fr:
        fr.write("qid\tpassage_id\tgrade\n")
        for split, ids, qrels in [("val", val_ids, val), ("test", sorted(test), test)]:
            for qid in ids:
                if qid not in qtext:
                    continue
                fq.write(json.dumps({"qid": qid, "text": qtext[qid], "type": "unknown", "split": split}) + "\n")
                for did, g in sorted(qrels[qid].items()):
                    fr.write(f"{qid}\t{did}\t{g}\n")
    return {"collection": name, "passages": len(passages), "val_queries": len(val_ids),
            "test_queries": len(test)}
