from contextweave.core import load_config
from contextweave.textproc import normalize, normalize_from_config
from contextweave.textproc.tokenizer import tokenize


def test_technical_tokens_kept():
    toks = tokenize("TF-IDF and 802.11 in C++ use a B+tree, k-means.")
    for t in ["tf-idf", "802.11", "c++", "b+tree", "k-means"]:
        assert t in toks


def test_normalize_stops_and_stems():
    terms = normalize("The running runs of the searches")
    assert "the" not in terms and "of" not in terms
    assert terms.count("run") == 2 and "search" in terms


def test_hyphen_parts_emitted():
    terms = normalize("tf-idf weighting")
    assert "tf-idf" in terms and "tf" in terms and "idf" in terms


def test_same_pipeline_for_queries_and_docs():
    cfg = load_config()
    assert normalize_from_config("Inverse Document Frequencies", cfg) == normalize("inverse document frequencies")
