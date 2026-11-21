from contextweave.controller.budget import Budget
from contextweave.controller.fusion import fuse, minmax
from contextweave.controller.voc import GainModel, choose_action
from contextweave.core.types import ScoredPassage
from tests.conftest import make_fixture_system


def sp(pid, score, m):
    return ScoredPassage(pid, pid.split("_")[0], score, m)


def test_fusion_formula_and_missing_items():
    lex = [sp("a_1", 10, "bm25"), sp("b_1", 5, "bm25"), sp("c_1", 0, "bm25")]
    den = [sp("b_1", 0.9, "dense"), sp("d_1", 0.5, "dense")]
    out = fuse({"bm25": lex, "dense": den}, alpha=0.6, k=10)
    s = {r.passage_id: r.score for r in out}
    assert s["a_1"] == 0.6 * 1.0 + 0.4 * 0.0
    assert abs(s["b_1"] - (0.6 * 0.5 + 0.4 * 1.0)) < 1e-12
    assert s["d_1"] == 0.0 + 0.4 * 0.0
    assert out[0].passage_id == "b_1" and out[0].explanation["alpha"] == 0.6
    assert minmax([sp("x_1", 3, "m")]) == {"x_1": 1.0}


def test_router_rules(fixture_system):
    r = fixture_system.engine().router
    assert r.first_method(fixture_system.query('"inverse document frequency"'))[0] == "bm25"
    assert r.first_method(fixture_system.query("semaphore AND lock"))[0] == "bm25"
    assert r.first_method(fixture_system.query("How do routers learn where to send packets?"))[0] == "dense"
    assert r.first_method(fixture_system.query("pagerank"))[0] == "bm25_champion"
    assert r.ladder("bm25_champion") == ["bm25", "dense", "expansion"]
    assert r.ladder("dense") == ["bm25", "expansion"]


def test_trace_shows_stop_and_escalation(fixture_system):
    eng = fixture_system.engine(tau=0.0, budget=100)
    stop = eng.search("semaphore sem_wait", k=5)
    actions = [t.action for t in stop.trace]
    assert actions[:3] == ["route", "retrieve", "confidence"] and actions[-1] == "stop"
    assert "confident" in stop.trace[-1].note and stop.total_cost == eng.costs["bm25_champion"]
    eng = fixture_system.engine(tau=1.01, budget=100)
    esc = eng.search("semaphore sem_wait", k=5)
    assert any(t.action == "escalate" for t in esc.trace)
    assert set(esc.extras["methods_run"]) == {"bm25_champion", "bm25", "dense", "expansion"}
    assert esc.results and esc.results[0].method == "fusion"


def test_budget_is_respected(fixture_system):
    eng = fixture_system.engine(tau=1.01)
    costs = eng.costs
    b = costs["bm25_champion"] + costs["bm25"]
    resp = eng.search("semaphore sem_wait", budget=b)
    assert resp.total_cost <= b + 1e-9
    assert "dense" not in resp.extras["methods_run"]
    assert any("skip dense" in t.note for t in resp.trace)


def test_fixed_modes(fixture_system):
    eng = fixture_system.engine()
    h = eng.search("semaphore", mode="hybrid")
    assert set(h.extras["methods_run"]) == {"bm25", "dense"}
    for m in ("bm25", "tfidf", "dense", "boolean", "bm25_champion"):
        assert eng.search("semaphore", mode=m).mode == m
    r = eng.search("semaphore", mode="route")
    assert r.trace[-1].note.startswith("routing only")


def test_domain_filter(fixture_system):
    eng = fixture_system.engine(tau=1.01, budget=100)
    resp = eng.search("algorithm", filters={"domains": ["OS"]})
    assert resp.results and all(fixture_system.store[r.passage_id].domain == "OS" for r in resp.results)


def test_qpp_signals_flag():
    s = make_fixture_system(features={"qpp_signals": True})
    resp = s.engine(tau=0.0).search("semaphore sem_wait")
    sig = [t for t in resp.trace if t.action == "confidence"][0].signals
    assert "nqc" in sig and "wig" in sig and 0 <= sig["wig"] <= 1


def test_voc_policy_picks_best_ratio(fixture_system):
    gm = GainModel([0.0, 0.5, 1.01], prior_strength=1.0).fit([
        {"action": "dense", "path": "lexical", "conf": 0.2, "gain": 0.20},
        {"action": "bm25", "path": "lexical", "conf": 0.2, "gain": 0.01},
        {"action": "expansion", "path": "lexical", "conf": 0.2, "gain": -0.05}])
    b = Budget(100, {"dense": 2.0, "bm25": 1.0, "expansion": 1.0})
    a, scored = choose_action(gm, ["bm25", "dense", "expansion"], "lexical", 0.2, b, lam=0.001)
    assert a == "dense" and scored["expansion"]["ratio"] < 0
    a, _ = choose_action(gm, ["expansion"], "lexical", 0.2, b, lam=0.001)
    assert a is None
    eng = fixture_system.engine(voc_model=gm, voc_lambda=0.001, budget=100)
    resp = eng.search("pagerank")
    assert any(t.note.startswith("voc") for t in resp.trace)
    assert "expansion" not in resp.extras["methods_run"]
