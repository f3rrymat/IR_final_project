"""Passage embeddings cached to disk + a FAISS IndexFlatIP (exact cosine search)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from contextweave.core.store import PassageStore

log = logging.getLogger(__name__)


def _fingerprint(store: PassageStore) -> str:
    h = hashlib.sha1()
    for p in store.passages:
        h.update(p.passage_id.encode())
        h.update(str(len(p.text)).encode())
    return h.hexdigest()


class FaissStore:
    def __init__(self, vectors: np.ndarray, passage_ids: list[str], doc_ids: list[str]):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.passage_ids, self.doc_ids = passage_ids, doc_ids
        self.pos = {pid: i for i, pid in enumerate(passage_ids)}
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @property
    def N(self) -> int:  # noqa: N802
        return len(self.passage_ids)

    def search(self, qvec: np.ndarray, k: int, candidate_ids: Optional[set[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, passage indices). With candidates, re-score exactly that set."""
        if candidate_ids is not None:
            idx = np.array(sorted(self.pos[c] for c in candidate_ids if c in self.pos), dtype=np.int64)
            if idx.size == 0:
                return np.zeros(0), np.zeros(0, dtype=np.int64)
            sims = self.vectors[idx] @ qvec
            order = np.argsort(-sims, kind="stable")[:k]
            return sims[order], idx[order]
        D, I = self.index.search(qvec.reshape(1, -1).astype(np.float32), min(k, self.N))  # noqa: N806
        keep = I[0] >= 0
        return D[0][keep], I[0][keep]

    def vector(self, pid: str) -> np.ndarray:
        return self.vectors[self.pos[pid]]

    @classmethod
    def load_or_build(cls, paths: dict, store: PassageStore, embedder, build_missing: bool = True) -> "FaissStore":
        emb_path, meta_path = Path(paths["embeddings"]), Path(paths["meta"])
        fp = _fingerprint(store)
        if emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp and meta.get("model") == embedder.model_name:
                return cls(np.load(emb_path), [p.passage_id for p in store], store.doc_of)
            log.warning("embedding cache is stale; rebuilding")
        if not build_missing:
            raise FileNotFoundError(emb_path)
        t = time.perf_counter()
        texts = [f"{p.title}. {p.text}" if p.domain not in {"scifact", "nfcorpus"} else p.text for p in store]
        vecs = embedder.encode(texts, show_progress=True)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, vecs)
        meta_path.write_text(json.dumps({"model": embedder.model_name, "n": len(texts), "dim": int(vecs.shape[1]),
                                         "fingerprint": fp, "seconds": round(time.perf_counter() - t, 1)}), encoding="utf-8", newline="")
        fs = cls(vecs, [p.passage_id for p in store], store.doc_of)
        faiss.write_index(fs.index, str(paths["faiss"]))
        return fs
