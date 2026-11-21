"""Composition root: loads (or builds) a collection's indexes and registers every
retriever by name.

This is the only module that imports concrete retriever classes. The controller,
the UI, the CLI and the experiments get everything through ``System`` and the
registry, so a team member can add a new retriever by registering it here (or
from their own module) without touching anything else.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional

from contextweave.core import registry
from contextweave.core.config import Config, load_config, set_seed
from contextweave.core.store import PassageStore
from contextweave.core.types import Query
from contextweave.indexing.boolean import BooleanRetriever
from contextweave.indexing.builder import build_index
from contextweave.indexing.inverted_index import InvertedIndex
from contextweave.lexical.bm25 import BM25Retriever, BM25Scorer
from contextweave.lexical.champion import ChampionRetriever
from contextweave.lexical.index_elimination import IndexEliminationRetriever
from contextweave.lexical.tfidf import TfidfRetriever
from contextweave.textproc.normalizer import normalize_from_config
from contextweave.controller.features import build_query

log = logging.getLogger(__name__)


@dataclass
class System:
    cfg: Config
    store: PassageStore
    index: InvertedIndex
    scorer: BM25Scorer
    normalize: Any
    semantic: bool = False
    embedder: Any = None
    vectors: Any = None
    extras: dict = field(default_factory=dict)
    # This system's own retrievers. The registry is global, so loading a second collection (e.g. switching
    # collections in the UI) replaces its entries; the controller uses these first so a cached System keeps
    # searching its own indexes. Empty -> the controller falls back to the global registry.
    retrievers: dict = field(default_factory=dict)

    def query(self, raw: str, filters: Optional[dict] = None) -> Query:
        return build_query(raw, self.normalize, self.index.idf, self.index.df, filters)

    def engine(self, **overrides):
        """An adaptive controller bound to this system (imported lazily)."""
        from contextweave.controller.engine import Controller

        return Controller(self, **overrides)


def collection_paths(cfg: Config, collection: Optional[str] = None) -> dict[str, Path]:
    c = collection or cfg["collection"]
    idx = Path(cfg["paths"]["indexes"]) / c
    return {"passages": Path(cfg["paths"]["processed"]) / c / "passages.jsonl", "index_dir": idx,
            "inverted": idx / "inverted.pkl", "embeddings": idx / "embeddings.npy",
            "faiss": idx / "faiss.index", "meta": idx / "embeddings_meta.json"}


def build_lexical_index(cfg: Config, collection: Optional[str] = None) -> InvertedIndex:
    paths = collection_paths(cfg, collection)
    store = PassageStore.from_jsonl(paths["passages"])
    t = time.perf_counter()
    norm = partial(normalize_from_config, cfg=cfg)
    index = build_index(store.passages, norm, cfg["indexing"]["block_size"])
    index.save(paths["inverted"])
    log.info("indexed %d passages, %d terms, %d postings in %.1fs", index.N, index.vocab_size,
             index.total_postings(), time.perf_counter() - t)
    return index


def load_system(cfg: Optional[Config] = None, collection: Optional[str] = None, semantic: bool = True,
                build_missing: bool = True) -> System:
    """Load a collection, build missing indexes, and register all retrievers."""
    cfg = cfg or load_config()
    if collection:
        cfg = Config({**cfg, "collection": collection})
    set_seed(cfg["seed"])
    paths = collection_paths(cfg)
    store = PassageStore.from_jsonl(paths["passages"])
    if paths["inverted"].exists():
        index = InvertedIndex.load(paths["inverted"])
    elif build_missing:
        index = build_lexical_index(cfg)
    else:
        raise FileNotFoundError(paths["inverted"])
    norm = partial(normalize_from_config, cfg=cfg)
    b = cfg["bm25"]
    scorer = BM25Scorer(index, b["k1"], b["b"], b.get("idf_variant", "lucene"))
    costs = cfg["costs"]
    ie = cfg["index_elimination"]
    system = System(cfg, store, index, scorer, norm)
    registry.clear()
    for r in (BooleanRetriever(index, norm, costs.get("boolean", 0.2)), TfidfRetriever(index, costs["tfidf"]),
              BM25Retriever(scorer, costs["bm25"]),
              IndexEliminationRetriever(scorer, ie["idf_threshold"], ie["min_terms_matched"], costs.get("bm25_ie", 0.6)),
              ChampionRetriever(scorer, cfg["champion"]["r"], ie["idf_threshold"], ie["min_terms_matched"],
                                costs["bm25_champion"])):
        _add(system, r)
    if semantic:
        attach_semantic(system, build_missing=build_missing)
    return system


def _add(system: System, retriever) -> None:
    """Give ``system`` its own reference to ``retriever`` and register it globally by name.

    ``system.retrievers`` is filled directly (never read back from the global registry), so two collections
    loaded at the same time - e.g. by two web requests - cannot end up with each other's retrievers.
    """
    system.retrievers[retriever.name] = retriever
    registry.register(retriever)


def attach_semantic(system: System, build_missing: bool = True) -> None:
    """Load the embedding model + vectors and register ``dense`` and ``expansion``."""
    from contextweave.semantic.dense_retriever import DenseRetriever
    from contextweave.semantic.embedder import Embedder
    from contextweave.semantic.expansion import ExpansionRetriever
    from contextweave.semantic.faiss_store import FaissStore

    cfg = system.cfg
    paths = collection_paths(cfg)
    d = cfg["dense"]
    embedder = Embedder(d["model"], d["batch_size"], d.get("max_seq_length", 256))
    vs = FaissStore.load_or_build(paths, system.store, embedder, build_missing=build_missing)
    system.embedder, system.vectors, system.semantic = embedder, vs, True
    costs = cfg["costs"]
    _add(system, DenseRetriever(vs, embedder, costs["dense"]))
    e = cfg["expansion"]
    _add(system, ExpansionRetriever(system.scorer, system.index, system.store, embedder, vs,
                                    system.normalize, cost=costs["expansion"], max_terms=e["max_terms"],
                                    weight=e["weight"], feedback_docs=e["feedback_docs"],
                                    candidate_terms=e["candidate_terms"],
                                    min_similarity=e["min_similarity"]))
