"""Sentence-Transformers embedder (fixed all-MiniLM-L6-v2, no training), CPU only."""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np

from contextweave.core.types import Query

log = logging.getLogger(__name__)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")  # transformers colours the report when stdout is a terminal


class _BenignLoadReportFilter(logging.Filter):
    """Drop transformers' "LOAD REPORT" warning when its only finding is an UNEXPECTED ``*.position_ids`` key.

    Checkpoints saved by older transformers (all-MiniLM-L6-v2 included) store BERT's ``position_ids`` buffer;
    transformers >= 5 makes it non-persistent and rebuilds it as ``arange(max_positions)``, so the stored copy is
    reported as unexpected. Both are identical (0..511) and no weight is lost. Any other finding (MISSING,
    MISMATCH, other UNEXPECTED keys) still gets through. Colour codes are stripped first: in a terminal the
    report reads "\x1b[...mUNEXPECTED\x1b[0m", not "UNEXPECTED".
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = _ANSI.sub("", record.getMessage())
        if "LOAD REPORT" not in msg:
            return True
        rows = [line.split("|") for line in msg.splitlines() if line.count("|") >= 2]
        findings = [(r[0].strip(), r[1].strip()) for r in rows if r[1].strip() and r[0].strip() != "Key"
                    and not set(r[0].strip()) <= set("-+")]
        return not findings or not all(s == "UNEXPECTED" and k.endswith("position_ids") for k, s in findings)


def _quiet_transformers() -> None:
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # "Loading weights: 100%|####|" on every start
        for name in ("transformers.modeling_utils", "transformers.utils.loading_report"):
            lg = logging.getLogger(name)
            if not any(isinstance(f, _BenignLoadReportFilter) for f in lg.filters):
                lg.addFilter(_BenignLoadReportFilter())
    except ImportError:  # pragma: no cover
        pass


class Embedder:
    """Lazy-loading wrapper. All vectors are L2-normalised float32, so inner product = cosine."""

    def __init__(self, model_name: str, batch_size: int = 64, max_seq_length: int = 256):
        self.model_name, self.batch_size, self.max_seq_length = model_name, batch_size, max_seq_length
        self._model = None
        self._term_cache: dict[str, np.ndarray] = {}

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(max(1, torch.get_num_threads()))
            _quiet_transformers()
            self._model = SentenceTransformer(self._local_path() or self.model_name, device="cpu")
            self._model.max_seq_length = self.max_seq_length
            log.info("loaded %s", self.model_name)
        return self._model

    def _local_path(self) -> Optional[str]:
        """Folder of the already-downloaded model, if any. Loading from a folder makes no Hub requests (faster
        start, works offline, no "unauthenticated requests" warning); the first run downloads by name."""
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(self.model_name, local_files_only=True)
        except Exception:
            return None

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return vecs.astype(np.float32)

    def encode_query(self, query: Query) -> np.ndarray:
        """Query vector, computed once per Query object and cached on it."""
        v = query.cache.get("qvec")
        if v is None:
            v = self.encode([query.raw])[0]
            query.cache["qvec"] = v
        return v

    def encode_terms(self, terms: Sequence[str]) -> np.ndarray:
        """Embeddings of single words (used by query expansion), cached across queries."""
        missing = [t for t in terms if t not in self._term_cache]
        if missing:
            for t, v in zip(missing, self.encode(missing)):
                self._term_cache[t] = v
        return np.stack([self._term_cache[t] for t in terms]) if terms else np.zeros((0, 1), np.float32)

    def similarity(self, query: Query, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        return self.encode(texts) @ self.encode_query(query)
