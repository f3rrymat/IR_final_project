import math

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from contextweave.core.store import PassageStore
from contextweave.core.types import Query
from contextweave.indexing.builder import build_index
from contextweave.lexical.bm25 import BM25Retriever, BM25Scorer
from contextweave.lexical.champion import ChampionRetriever
from contextweave.lexical.explain import term_rows
from contextweave.lexical.index_elimination import IndexEliminationRetriever, eliminate_terms
from contextweave.lexical.tfidf import TfidfRetriever
from contextweave.textproc import normalize
from tests.conftest import FIXTURES

QUERIES = ["semaphore lock shared data", "inverse document frequency weighting", "tcp congestion slow start",
           "shortest paths negative edges", "the page importance link graph", "query expansion feedback"]


@pytest.fixture(scope="module")
def setup():
    store = PassageStore.from_jsonl(FIXTURES / "passages.jsonl")
    idx = build_index(store.passages, normalize)
    return store, idx


def q(text):
    return Query(text, normalize(text))


def test_bm25_matches_rank_bm25(setup):
    store, idx = setup
    corpus = [normalize(p.text) for p in store.passages]
    ref = BM25Okapi(corpus, k1=1.2, b=0.75, epsilon=0.25)
    scorer = BM25Scorer(idx, 1.2, 0.75, idf_variant="okapi")
    for text in QUERIES:
        terms = normalize(text)
        expected = ref.get_scores(terms)
        ours, _ = scorer.accumulate(scorer.query_weights(Query(text, terms)))
        assert np.allclose(ours, expected, atol=1e-9), text


def test_bm25_lucene_variant_ranks_and_explains(setup):
    store, idx = setup
    r = BM25Retriever(BM25Scorer(idx, 1.2, 0.75))
    res = r.search(q("semaphore sem_wait"), k=3)
    assert store[res[0].passage_id].title == "Semaphores"
    rows = term_rows(res[0])
    assert sum(x["score"] for x in rows) == pytest.approx(res[0].score, rel=1e-3)
    assert r.last_stats["passages_scored"] >= 1
    only = {store.passages[0].passage_id, store.passages[1].passage_id}
    assert {x.passage_id for x in r.search(q("idf bm25"), 10, candidate_ids=only)} <= only


def test_tfidf_cosine_matches_bruteforce(setup):
    store, idx = setup
    r = TfidfRetriever(idx)
    query = q("inverse document frequency weighting")
    res = r.search(query, k=5)
    # brute force cosine with the same weighting
    vocab = list(idx.postings)
    def vec(terms):
        v = np.zeros(len(vocab))
        for t in set(terms):
            if t in idx.postings:
                v[vocab.index(t)] = (1 + math.log(terms.count(t))) * idx.idf(t)
        return v
    qv = vec(query.terms)
    bf = []
    for p in store.passages:
        dv = vec(normalize(p.text))
        bf.append(float(qv @ dv / (np.linalg.norm(qv) * (np.linalg.norm(dv) or 1))))
    best = int(np.argmax(bf))
    assert res[0].passage_id == store.passages[best].passage_id
    assert res[0].score == pytest.approx(bf[best], rel=1e-6)


def test_champion_scores_only_candidates(setup):
    store, idx = setup
    scorer = BM25Scorer(idx)
    full = BM25Retriever(scorer).search(q("semaphore lock"), 5)
    champ = ChampionRetriever(scorer, r=2, idf_threshold=0.0, min_terms_matched=1)
    res = champ.search(q("semaphore lock"), 5)
    assert champ.last_stats["passages_scored"] <= 4  # 2 champions per term
    assert res[0].passage_id == full[0].passage_id
    for t, docs in champ.champions.items():
        assert len(docs) <= 2


def test_index_elimination_drops_low_idf_terms(setup):
    store, idx = setup
    scorer = BM25Scorer(idx)
    kept, dropped = eliminate_terms(scorer, {"semaphor": 1.0, "zzz": 1.0}, idf_threshold=0.5)
    assert "semaphor" in kept and "zzz" in dropped
    common = max(idx.postings, key=idx.df)
    kept, dropped = eliminate_terms(scorer, {common: 1.0, "semaphor": 1.0}, idf_threshold=idx.idf(common) + 0.01)
    assert common in dropped
    r = IndexEliminationRetriever(scorer, idf_threshold=0.5, min_terms_matched=2)
    r.search(q("semaphore lock binary"), 5)
    assert r.last_stats["min_match"] in (1, 2)
