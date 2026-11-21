import numpy as np
import pytest

from contextweave.core.store import PassageStore
from contextweave.indexing.boolean import BooleanParser, difference, intersect, union
from contextweave.indexing.builder import build_index
from contextweave.indexing.inverted_index import InvertedIndex
from contextweave.indexing.snippets import make_snippet
from contextweave.textproc import normalize
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def store():
    return PassageStore.from_jsonl(FIXTURES / "passages.jsonl")


def test_blocked_build_equals_single_block(store):
    a = build_index(store.passages, normalize, block_size=3)
    b = build_index(store.passages, normalize, block_size=10_000)
    assert a.postings.keys() == b.postings.keys()
    for t in a.postings:
        assert np.array_equal(a.postings[t].docs, b.postings[t].docs)
        assert np.array_equal(a.postings[t].tfs, b.postings[t].tfs)
        assert list(a.postings[t].docs) == sorted(a.postings[t].docs)


def test_index_statistics_and_persistence(store, tmp_path):
    idx = build_index(store.passages, normalize, block_size=5)
    assert idx.N == len(store) and idx.lengths.sum() == idx.total_postings() or idx.lengths.sum() >= idx.total_postings()
    assert idx.df("semaphor") >= 1 and idx.idf("zzzz") == 0.0
    idx.save(tmp_path / "i.pkl")
    again = InvertedIndex.load(tmp_path / "i.pkl")
    assert again.vocab_size == idx.vocab_size and again.avgdl == pytest.approx(idx.avgdl)


def test_merge_algorithms():
    assert intersect([1, 3, 5, 7], [3, 4, 5]) == [3, 5]
    assert union([1, 3], [2, 3, 9]) == [1, 2, 3, 9]
    assert difference([1, 2, 3, 4], [2, 4]) == [1, 3]


def test_boolean_queries(store):
    idx = build_index(store.passages, normalize)
    bp = BooleanParser(idx, normalize)
    sem, _ = bp.evaluate("semaphore")
    lock, _ = bp.evaluate("lock")
    both, _ = bp.evaluate("semaphore AND lock")
    either, _ = bp.evaluate("semaphore OR lock")
    notlock, _ = bp.evaluate("semaphore AND NOT lock")
    assert set(both) == set(sem) & set(lock)
    assert set(either) == set(sem) | set(lock)
    assert set(notlock) == set(sem) - set(lock)
    grouped, terms = bp.evaluate("(deadlock OR semaphore) AND lock")
    assert set(grouped) == (set(bp.evaluate("deadlock")[0]) | set(sem)) & set(lock)
    assert "deadlock" in terms


def test_snippet_highlights_best_window():
    text = "alpha beta gamma " * 20 + "Retrieval of retrieved documents " + "delta " * 30
    s = make_snippet(text, normalize("retrieval"), normalize, window=10)
    assert s["hits"] == 2 and s["html"].count("<mark>") == 2
