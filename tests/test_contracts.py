"""Contract tests: shared types, Retriever protocol and registry behave as specified."""
from __future__ import annotations

import dataclasses

import pytest

from contextweave.core import (Passage, Query, Retriever, ScoredPassage, SearchResponse,
                               TraceStep, finalize, load_config, registry)


class Dummy:
    name = "dummy"
    cost = 0.5

    def search(self, query, k, candidate_ids=None):
        res = [ScoredPassage(f"p{i}", "d", float(i), self.name) for i in range(5)]
        if candidate_ids is not None:
            res = [r for r in res if r.passage_id in candidate_ids]
        return finalize(res, k)


def test_passage_is_frozen():
    p = Passage("d_p0", "d", "text", "t", "s", 1, "IR", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.text = "x"  # type: ignore[misc]


def test_type_fields_match_spec():
    names = lambda c: [f.name for f in dataclasses.fields(c)]  # noqa: E731
    assert names(Passage) == ["passage_id", "doc_id", "text", "title", "source", "page", "domain", "n_tokens"]
    assert names(Query)[:4] == ["raw", "terms", "features", "filters"]
    assert names(ScoredPassage) == ["passage_id", "doc_id", "score", "method", "rank", "norm_score", "explanation"]
    assert names(TraceStep) == ["action", "method", "cost", "budget_left", "confidence", "signals", "latency_ms", "note"]
    assert names(SearchResponse)[:6] == ["query", "results", "trace", "total_cost", "total_latency_ms", "mode"]


def test_registry_roundtrip():
    registry.clear()
    d = Dummy()
    assert isinstance(d, Retriever)
    registry.register(d)
    assert registry.available() == ["dummy"]
    assert registry.get("dummy") is d
    with pytest.raises(KeyError):
        registry.get("missing")


def test_registry_rejects_non_retriever():
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


def test_finalize_sorts_and_ranks():
    out = Dummy().search(Query("q", ["q"]), k=3)
    assert [r.passage_id for r in out] == ["p4", "p3", "p2"]
    assert [r.rank for r in out] == [1, 2, 3]
    out = Dummy().search(Query("q", ["q"]), k=10, candidate_ids={"p1", "p2"})
    assert [r.passage_id for r in out] == ["p2", "p1"]


def test_config_loads_with_required_sections():
    cfg = load_config()
    for key in ["seed", "chunking", "text", "bm25", "champion", "index_elimination", "dense",
                "expansion", "fusion", "controller", "costs", "features"]:
        assert key in cfg, key
    assert cfg.bm25.k1 == 1.2 and cfg.controller.budget > 0
    # every novel extension is off by default -> baseline == specification
    assert not any(cfg["features"].values())
    over = load_config(overrides={"bm25": {"k1": 2.0}})
    assert over.bm25.k1 == 2.0 and over.bm25.b == 0.75
