"""Inverted index: term -> (document frequency, postings of (passage index, tf)).

Postings are stored as two aligned numpy arrays sorted by passage index, which
keeps memory low and lets the scorers do term-at-a-time accumulation quickly.
Collection statistics (N, passage lengths, average length) are stored for scoring.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Postings:
    """Postings list of one term: passage indices (sorted) and term frequencies."""

    docs: np.ndarray  # int32, sorted ascending
    tfs: np.ndarray  # int32

    @property
    def df(self) -> int:
        return int(self.docs.shape[0])


@dataclass
class InvertedIndex:
    passage_ids: list[str]
    doc_ids: list[str]
    lengths: np.ndarray  # int32 number of index terms per passage
    postings: dict[str, Postings] = field(default_factory=dict)

    # -- collection statistics -------------------------------------------------
    @property
    def N(self) -> int:  # noqa: N802 (IR notation)
        return len(self.passage_ids)

    @property
    def avgdl(self) -> float:
        return float(self.lengths.mean()) if self.N else 0.0

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        return 0 if p is None else p.df

    def idf(self, term: str) -> float:
        """Plain IDF, log(N / df); 0 for unknown terms. Used by TF-IDF and query features."""
        d = self.df(term)
        return math.log(self.N / d) if d else 0.0

    def get(self, term: str) -> Postings | None:
        return self.postings.get(term)

    @property
    def pos(self) -> dict[str, int]:
        """passage_id -> integer index (built lazily, cached)."""
        cache = self.__dict__.get("_pos")
        if cache is None:
            cache = {pid: i for i, pid in enumerate(self.passage_ids)}
            self.__dict__["_pos"] = cache
        return cache

    def mask(self, candidate_ids) -> "np.ndarray | None":
        """Boolean mask over passages for a candidate id set (None -> no restriction)."""
        if candidate_ids is None:
            return None
        m = np.zeros(self.N, dtype=bool)
        pos = self.pos
        m[[pos[c] for c in candidate_ids if c in pos]] = True
        return m

    def total_postings(self) -> int:
        return int(sum(p.df for p in self.postings.values()))

    # -- persistence -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "InvertedIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
