import numpy as np

from contextweave.core import registry
from tests.conftest import FakeEmbedder


def test_dense_search_and_candidate_rescoring(fixture_system):
    s = fixture_system
    q = s.query("semaphore sem_wait sem_post lock")
    res = registry.get("dense").search(q, 5)
    assert s.store[res[0].passage_id].title == "Semaphores"
    assert all(-1.0001 <= r.score <= 1.0001 for r in res)
    cand = {res[2].passage_id, res[3].passage_id}
    sub = registry.get("dense").search(q, 5, candidate_ids=cand)
    assert {r.passage_id for r in sub} == cand
    # exact cosine: FAISS flat IP equals numpy dot on normalised vectors
    v = s.vectors.vector(res[0].passage_id)
    assert np.isclose(res[0].score, float(v @ q.cache["qvec"]), atol=1e-5)


def test_query_vector_computed_once(fixture_system):
    q = fixture_system.query("deadlock")
    registry.get("dense").search(q, 3)
    first = q.cache["qvec"]
    registry.get("dense").search(q, 3)
    assert q.cache["qvec"] is first


def test_expansion_adds_terms_and_reports_them(fixture_system):
    s = fixture_system
    exp = registry.get("expansion")
    q = s.query("semaphore")
    res = exp.search(q, 5)
    added = exp.last_stats["added_terms"]
    assert 0 < len(added) <= 5
    assert all(a["term"] not in q.terms for a in added)
    assert res and "added_terms" in res[0].explanation


def test_expansion_drift_filter_can_reject_everything(fixture_system):
    exp = registry.get("expansion")
    exp.min_sim = 2.0  # cosine can never reach 2 -> nothing added
    q = fixture_system.query("semaphore")
    exp.search(q, 5)
    assert exp.last_stats["added_terms"] == []
    exp.min_sim = -1.0


def test_fake_embedder_is_normalised():
    e = FakeEmbedder()
    v = e.encode(["bm25 ranking", "tcp congestion"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def _report(rows):
    import logging

    table = "Key                     | Status     |  | \n------------------------+------------+--+-\n"
    table += "".join(f"{k} | {s} |  | \n" for k, s in rows)
    return logging.LogRecord("transformers.modeling_utils", logging.WARNING, "", 0,
                             "BertModel LOAD REPORT from: x\n" + table, None, None)


def test_benign_position_ids_load_report_is_silenced_but_real_problems_are_not():
    import logging

    from contextweave.semantic.embedder import _BenignLoadReportFilter

    f = _BenignLoadReportFilter()
    assert not f.filter(_report([("embeddings.position_ids", "UNEXPECTED")]))      # transformers>=5 + old ckpt
    assert f.filter(_report([("encoder.layer.0.attention.self.query.weight", "MISSING")]))
    assert f.filter(_report([("embeddings.position_ids", "UNEXPECTED"), ("pooler.dense.weight", "MISSING")]))
    assert f.filter(_report([("classifier.weight", "UNEXPECTED")]))
    assert f.filter(logging.LogRecord("x", logging.WARNING, "", 0, "some other warning", None, None))


def test_load_report_filter_handles_terminal_colours():
    """In a terminal transformers wraps names and statuses in ANSI colour codes (python main.py serve)."""
    from contextweave.semantic.embedder import _BenignLoadReportFilter

    f = _BenignLoadReportFilter()
    orange, bold, reset = "\x1b[38;5;208m", "\x1b[1m", "\x1b[0m"
    rec = _report([("embeddings.position_ids", f"{orange}UNEXPECTED{reset}")])
    rec.msg = rec.msg.replace("BertModel LOAD REPORT", f"{bold}BertModel LOAD REPORT{reset}")
    assert not f.filter(rec)
    assert f.filter(_report([("pooler.dense.weight", f"\x1b[31mMISSING{reset}")]))
