"""Blocked index construction with a k-way merge (SPIMI-style).

1. Read passages in blocks of ``block_size``.
2. For each block build an in-memory dictionary term -> [(passage_idx, tf)],
   sort the terms and write the block to disk as a sorted run.
3. k-way merge the sorted runs (heap on the term) into the final index.

This is the textbook "single-pass in-memory indexing + merge" algorithm; for a
laptop-sized corpus it would also fit in memory, but the blocks show the method.
"""
from __future__ import annotations

import heapq
import logging
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from contextweave.core.types import Passage
from contextweave.indexing.inverted_index import InvertedIndex, Postings

log = logging.getLogger(__name__)

Run = list[tuple[str, list[tuple[int, int]]]]


def _write_run(block: dict[str, list[tuple[int, int]]], path: Path) -> None:
    with open(path, "wb") as f:
        for term in sorted(block):
            pickle.dump((term, block[term]), f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_run(path: Path) -> Iterator[tuple[str, list[tuple[int, int]]]]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def build_index(passages: Sequence[Passage], normalize: Callable[[str], list[str]],
                block_size: int = 2000, tmp_dir: str | Path | None = None) -> InvertedIndex:
    """Build an inverted index over ``passages`` using blocked construction + merge."""
    lengths = np.zeros(len(passages), dtype=np.int32)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        runs: list[Path] = []
        block: dict[str, list[tuple[int, int]]] = {}
        for i, p in enumerate(passages):
            terms = normalize(p.text)
            lengths[i] = len(terms)
            for term, tf in Counter(terms).items():
                block.setdefault(term, []).append((i, tf))
            if (i + 1) % block_size == 0:
                runs.append(Path(td) / f"run{len(runs):04d}.pkl")
                _write_run(block, runs[-1])
                block = {}
        if block:
            runs.append(Path(td) / f"run{len(runs):04d}.pkl")
            _write_run(block, runs[-1])
        log.info("wrote %d sorted runs; merging", len(runs))
        postings = merge_runs([_read_run(r) for r in runs])
    return InvertedIndex([p.passage_id for p in passages], [p.doc_id for p in passages], lengths, postings)


def merge_runs(runs: list[Iterator[tuple[str, list[tuple[int, int]]]]]) -> dict[str, Postings]:
    """k-way merge of sorted runs. Runs come from increasing passage ranges, so the
    postings of a term can be concatenated in run order and stay sorted."""
    heap: list[tuple[str, int, list[tuple[int, int]]]] = []
    for ri, it in enumerate(runs):
        first = next(it, None)
        if first is not None:
            heapq.heappush(heap, (first[0], ri, first[1]))
    out: dict[str, Postings] = {}
    cur_term: str | None = None
    cur: list[tuple[int, int]] = []
    while heap:
        term, ri, plist = heapq.heappop(heap)
        if term != cur_term:
            if cur_term is not None:
                out[cur_term] = _to_postings(cur)
            cur_term, cur = term, []
        cur.extend(plist)
        nxt = next(runs[ri], None)
        if nxt is not None:
            heapq.heappush(heap, (nxt[0], ri, nxt[1]))
    if cur_term is not None:
        out[cur_term] = _to_postings(cur)
    return out


def _to_postings(plist: list[tuple[int, int]]) -> Postings:
    arr = np.asarray(plist, dtype=np.int32).reshape(-1, 2)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    return Postings(docs=arr[:, 0].copy(), tfs=arr[:, 1].copy())
