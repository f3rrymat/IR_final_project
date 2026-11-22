"""Web API tests (FastAPI TestClient) on the small fixture corpus - no model download, no built indexes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ROOT, make_fixture_system


def _own_retrievers(system):
    """Give a fixture System its own retrievers, as load_system() does."""
    from contextweave.core import registry

    system.retrievers = {n: registry.get(n) for n in registry.available()}
    return system


@pytest.fixture
def client():
    from api.server import create_app

    sys_ = _own_retrievers(make_fixture_system())
    app = create_app(loader=lambda c: sys_, warm=False, is_built=lambda c: c == "cs")
    with TestClient(app) as c:
        yield c


def test_health_and_collections(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["built"] == ["cs"]
    d = client.get("/api/collections").json()
    assert d["default"] == "cs" and "auto" in d["modes"]
    cs = next(c for c in d["collections"] if c["id"] == "cs")
    assert cs["ready"] and cs["judging"] and cs["examples"]
    assert {"tau", "budget"} <= set(cs["defaults"]) and cs["max_budget"] > cs["costs"]["bm25"]
    assert not next(c for c in d["collections"] if c["id"] == "scifact")["ready"]


def test_search_auto_returns_trace_results_and_explanations(client):
    r = client.post("/api/search", json={"collection": "cs", "query": "semaphore sem_wait sem_post lock",
                                         "budget": 50, "tau": 0.99})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == "auto" and d["trace"][0]["action"] == "route" and d["trace"][-1]["action"] == "stop"
    assert d["results"] and d["results"][0]["rank"] == 1 and d["results"][0]["title"]
    assert d["total_cost"] <= 50 + max(t["cost"] for t in d["trace"])  # budget is respected step by step
    assert [t["term"] for t in d["query"]["terms"]][:1] == ["semaphor"]
    top = d["results"][0]
    assert any(hit for _, hit in top["snippet"]["segments"]), "query terms are highlighted"
    assert top["why"]["terms"] or top["why"]["cosine"] is not None
    assert set(d["methods_run"]) <= set(d["lists"]) and d["lists"][d["methods_run"][0]]["top"]
    json.dumps(d)  # plain JSON (no NaN / numpy types)


@pytest.mark.parametrize("mode", ["bm25", "tfidf", "dense", "hybrid", "boolean", "bm25_champion", "route"])
def test_search_every_mode(client, mode):
    r = client.post("/api/search", json={"collection": "cs", "query": "tf-idf weight", "mode": mode})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == mode
    if mode in ("bm25", "tfidf", "dense"):
        assert d["methods_run"] == [mode] and all(x["method"] == mode for x in d["results"])


def test_search_rejects_bad_input(client):
    assert client.post("/api/search", json={"collection": "cs", "query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/api/search", json={"collection": "cs", "query": ""}).status_code == 422
    assert client.post("/api/search", json={"collection": "cs", "query": "x", "tau": 2}).status_code == 422
    assert client.post("/api/search", json={"collection": "nope", "query": "x"}).status_code == 404
    r = client.post("/api/search", json={"collection": "scifact", "query": "x"})
    assert r.status_code == 409 and "build-index" in r.json()["detail"]


def test_passage_endpoint(client):
    d = client.get("/api/collections/cs/passages/fx00_p0001").json()
    assert d["title"] == "BM25" and "k1" in d["text"]
    assert client.get("/api/collections/cs/passages/nope").status_code == 404


def test_evaluation_reads_saved_results(client):
    d = client.get("/api/evaluation/scifact").json()
    assert "test" in d["available"]
    r = d["result"]
    assert r["labelled"] and r["split"] == "test" and r["n_queries"] == 300
    s7 = next(s for s in r["systems"] if s["id"] == "S7")
    assert 0 < s7["metrics"]["NDCG@10"] < 1 and s7["mean_cost"] > 0
    assert r["sweep"] and r["risk_curve"] and r["significance"]
    json.dumps(d)
    cs = client.get("/api/evaluation/cs").json()["result"]
    assert cs is not None and not cs["labelled"]  # cs awaits team labels: costs only
    assert client.get("/api/evaluation/nope").status_code == 404


def test_judging_flow(client, tmp_path, monkeypatch):
    from contextweave.evaluation import benchmark as bench

    monkeypatch.setattr(bench, "bench_dir", lambda c: tmp_path)
    assert client.get("/api/judge/cs").json()["pool_ready"] is False
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"qid": "q1", "text": "what is bm25", "type": "semantic", "split": "test", "domain": "IR"}) + "\n",
        encoding="utf-8")
    (tmp_path / "pool.json").write_text(json.dumps({"q1": ["fx00_p0001", "fx00_p0000"]}), encoding="utf-8")

    st = client.get("/api/judge/cs").json()
    assert st["pool_ready"] and (st["done"], st["total"]) == (0, 2)
    item = client.get("/api/judge/cs/item").json()["item"]
    assert item["qid"] == "q1" and item["passage"]["passage_id"] == "fx00_p0001" and item["grade"] is None

    st = client.post("/api/judge/cs", json={"qid": "q1", "passage_id": "fx00_p0001", "grade": 2,
                                            "annotator": "Test Person"}).json()
    assert (st["done"], st["total"]) == (1, 2)
    assert client.get("/api/judge/cs/item").json()["item"]["passage"]["passage_id"] == "fx00_p0000"
    assert (tmp_path / "qrels.tsv").read_text(encoding="utf-8").splitlines()[1] == "q1\tfx00_p0001\t2"
    assert (tmp_path / "judgements_test_person.tsv").exists()
    regrade = client.get("/api/judge/cs/item", params={"qid": "q1", "passage_id": "fx00_p0001"}).json()["item"]
    assert regrade["grade"] == 2
    assert client.post("/api/judge/cs", json={"qid": "q1", "passage_id": "nope", "grade": 1}).status_code == 404
    assert client.post("/api/judge/cs", json={"qid": "q1", "passage_id": "fx00_p0000", "grade": 3}).status_code == 422


def test_each_collection_keeps_its_own_retrievers(tmp_path):
    """Regression: loading collection B used to make searches on a cached collection A hit B's index."""
    from api.server import create_app

    a = _own_retrievers(make_fixture_system())
    ids_a = {p.passage_id for p in a.store}
    other = tmp_path / "other.jsonl"  # same texts, different passage ids
    other.write_text("".join(json.dumps({**p.to_dict(), "passage_id": "other_" + p.passage_id}) + "\n"
                             for p in a.store), encoding="utf-8")
    b = _own_retrievers(make_fixture_system(passages=other))  # replaces every entry of the global registry
    systems = {"cs": a, "scifact": b}
    app = create_app(loader=systems.__getitem__, warm=False, is_built=lambda c: c in systems)
    with TestClient(app) as c:
        c.post("/api/search", json={"collection": "scifact", "query": "bm25", "mode": "bm25"})  # load B second
        d = c.post("/api/search", json={"collection": "cs", "query": "bm25 ranking", "mode": "auto"}).json()
    assert d["results"] and {r["passage_id"] for r in d["results"]} <= ids_a


def test_ui_routes(client):
    r = client.get("/")
    assert r.status_code == 200 and "<" in r.text  # built UI, or the "build the UI" page
    assert client.get("/judge").status_code == 200  # client-side route falls back to index.html
    assert client.get("/api/nope").status_code == 404
    if (ROOT / "web" / "dist" / "index.html").exists():
        assert 'id="root"' in r.text


def test_snippet_segments_keep_spaces_outside_highlights():
    from contextweave.indexing.snippets import make_snippet, snippet_segments
    from contextweave.textproc import normalize

    text = "Okapi BM25 ranks documents; the BM25 function saturates term frequency with k1."
    s = snippet_segments(text, {"bm25", "frequenc"}, normalize, window=8)
    joined = "".join(t for t, _ in s["segments"])
    assert joined == " ".join(text.split()[:8]) and s["after"] and not s["before"]  # first 8 words win the tie
    assert [t for t, hit in s["segments"] if hit] == ["BM25", "BM25"]
    assert all(t == t.strip() for t, hit in s["segments"] if hit)
    assert s["hits"] == make_snippet(text, {"bm25", "frequenc"}, normalize, window=8)["hits"]


def _keys(x):
    """Recursive key structure (dict keys only), so the comparison ignores values that legitimately vary."""
    if isinstance(x, dict):
        return {k: _keys(v) for k, v in x.items()}
    if isinstance(x, list):
        merged: dict = {}
        for v in x:
            k = _keys(v)
            if isinstance(k, dict):
                merged.update(k)
        return [merged] if merged else []
    return None


def test_ui_fixtures_match_the_api():
    """web/src/test/fixtures/*.json feed the UI unit tests; they must have exactly the API's shape.
    If this fails after an API change, run: python -m tests.make_ui_fixtures"""
    from tests.make_ui_fixtures import OUT, build

    fresh = build()
    for name, data in fresh.items():
        f = OUT / f"{name}.json"
        assert f.exists(), f"missing {f.name} - run python -m tests.make_ui_fixtures"
        saved = json.loads(f.read_text(encoding="utf-8"))
        assert _keys(saved) == _keys(data), f"{f.name} is out of date - run python -m tests.make_ui_fixtures"


def test_collections_loaded_concurrently_keep_their_own_retrievers(tmp_path):
    """Regression: the server warms one collection in a background thread while a request loads another.
    Both used to clear/fill the one global registry, so a system could end up with the other's retrievers."""
    import threading

    from contextweave.core.config import load_config
    from contextweave.system import load_system
    from tests.conftest import FIXTURES

    src = (FIXTURES / "passages.jsonl").read_text(encoding="utf-8").splitlines()
    for name in ("alpha", "beta"):
        d = tmp_path / "processed" / name
        d.mkdir(parents=True)
        (d / "passages.jsonl").write_text("".join(
            json.dumps({**json.loads(line), "passage_id": f"{name}_{json.loads(line)['passage_id']}"}) + "\n"
            for line in src), encoding="utf-8")
    cfg = load_config(overrides={"paths": {"processed": str(tmp_path / "processed"),
                                           "indexes": str(tmp_path / "indexes")}})
    for _ in range(3):
        out = {}
        ts = [threading.Thread(target=lambda n=n: out.__setitem__(n, load_system(cfg, collection=n, semantic=False)))
              for n in ("alpha", "beta")]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for name, system in out.items():
            assert set(system.retrievers) >= {"bm25", "tfidf", "boolean", "bm25_champion", "bm25_ie"}
            q = system.query("semaphore lock")
            for r in system.retrievers.values():
                assert all(x.passage_id.startswith(name + "_") for x in r.search(q, 5)), (name, r.name)
