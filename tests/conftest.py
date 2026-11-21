"""Shared pytest fixtures: the small fixture corpus in tests/fixtures/."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


import hashlib  # noqa: E402

import numpy as np  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-stems hashing embedder: lets semantic code be tested
    without downloading a model. Texts sharing stems get high cosine."""

    model_name = "fake-hash-embedder"
    dim = 64

    def __init__(self):
        from contextweave.textproc import normalize

        self._norm = normalize
        self._term_cache = {}

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in self._norm(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
            v[(h // 7) % self.dim] += 0.5
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, show_progress=False):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_query(self, query):
        if "qvec" not in query.cache:
            query.cache["qvec"] = self._vec(query.raw)
        return query.cache["qvec"]

    def encode_terms(self, terms):
        return self.encode(list(terms)) if terms else np.zeros((0, self.dim), np.float32)


def make_fixture_system(features=None, passages=None):
    """A full System over the fixture corpus with the fake embedder registered."""
    from functools import partial

    from contextweave.core import registry
    from contextweave.core.config import load_config
    from contextweave.core.store import PassageStore
    from contextweave.indexing.boolean import BooleanRetriever
    from contextweave.indexing.builder import build_index
    from contextweave.lexical.bm25 import BM25Retriever, BM25Scorer
    from contextweave.lexical.champion import ChampionRetriever
    from contextweave.lexical.index_elimination import IndexEliminationRetriever
    from contextweave.lexical.tfidf import TfidfRetriever
    from contextweave.semantic.dense_retriever import DenseRetriever
    from contextweave.semantic.expansion import ExpansionRetriever
    from contextweave.semantic.faiss_store import FaissStore
    from contextweave.system import System
    from contextweave.textproc.normalizer import normalize_from_config

    cfg = load_config(overrides={"collection": "fixture", "features": features or {},
                                 "retrieval": {"k": 5, "depth": 20}, "champion": {"r": 5},
                                 "index_elimination": {"idf_threshold": 0.5, "min_terms_matched": 1}})
    store = PassageStore.from_jsonl(passages or FIXTURES / "passages.jsonl")
    norm = partial(normalize_from_config, cfg=cfg)
    index = build_index(store.passages, norm)
    scorer = BM25Scorer(index)
    registry.clear()
    registry.register(BooleanRetriever(index, norm))
    registry.register(TfidfRetriever(index))
    registry.register(BM25Retriever(scorer))
    registry.register(IndexEliminationRetriever(scorer, 0.5, 1))
    registry.register(ChampionRetriever(scorer, 5, 0.5, 1))
    emb = FakeEmbedder()
    vs = FaissStore(emb.encode([p.text for p in store]), [p.passage_id for p in store], store.doc_of)
    registry.register(DenseRetriever(vs, emb))
    registry.register(ExpansionRetriever(scorer, index, store, emb, vs, norm, min_similarity=-1.0))
    return System(cfg, store, index, scorer, norm, semantic=True, embedder=emb, vectors=vs)


@pytest.fixture
def fixture_system():
    return make_fixture_system()
