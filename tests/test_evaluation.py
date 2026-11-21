import random

import ir_measures
import pytest
from ir_measures import AP, P, R, RR, nDCG

from contextweave.evaluation.metrics import aggregate, cohen_kappa, evaluate_run, ndcg_at
from contextweave.evaluation.significance import holm, paired_tests
from contextweave.controller.risk import binomial_pvalue, ltt_select


def _synthetic(seed=1, n_q=25, n_docs=60):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(n_q):
        qid = f"q{q}"
        docs = [f"d{i}" for i in range(n_docs)]
        qrels[qid] = {d: rng.choice([0, 0, 0, 1, 2]) for d in rng.sample(docs, 20)}
        if not any(g > 0 for g in qrels[qid].values()):
            qrels[qid][docs[0]] = 1
        run[qid] = rng.sample(docs, 40)
    return qrels, run


def test_metrics_match_ir_measures():
    qrels, run = _synthetic()
    ours = aggregate(evaluate_run(run, qrels))
    irq = [ir_measures.Qrel(q, d, g) for q, ds in qrels.items() for d, g in ds.items()]
    irr = [ir_measures.ScoredDoc(q, d, float(len(ds) - i)) for q, ds in run.items() for i, d in enumerate(ds)]
    ref = ir_measures.calc_aggregate([P @ 5, R @ 10, AP, nDCG @ 10, RR], irq, irr)
    assert ours["P@5"] == pytest.approx(ref[P @ 5], abs=1e-9)
    assert ours["Recall@10"] == pytest.approx(ref[R @ 10], abs=1e-9)
    assert ours["MAP"] == pytest.approx(ref[AP], abs=1e-9)
    assert ours["NDCG@10"] == pytest.approx(ref[nDCG @ 10], abs=1e-9)
    assert ours["MRR"] == pytest.approx(ref[RR], abs=1e-9)


def test_ndcg_perfect_and_empty():
    rels = {"a": 2, "b": 1}
    assert ndcg_at(["a", "b"], rels, 10) == pytest.approx(1.0)
    assert ndcg_at(["x"], rels, 10) == 0.0


def test_cohen_kappa():
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    k = cohen_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    assert -1 <= k < 1


def test_significance_and_holm():
    out = paired_tests([0.5, 0.6, 0.7, 0.8, 0.9] * 4, [0.4, 0.5, 0.6, 0.7, 0.85] * 4)
    assert out["mean_diff"] > 0 and out["t_p"] < 0.05 and out["ci95"][0] > 0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] >= adj["c"]


def test_learn_then_test_selection():
    # conservative taus fail rarely; aggressive ones fail often
    losses = {0.9: [0] * 95 + [1] * 5, 0.7: [0] * 90 + [1] * 10, 0.5: [0] * 60 + [1] * 40}
    res = ltt_select(losses, {0.9: 3, 0.7: 2, 0.5: 1}, delta=0.2, eta=0.1, order=[0.9, 0.7, 0.5])
    assert res["tau"] == 0.7 and res["certified"]
    assert binomial_pvalue(40, 100, 0.2) > 0.1
