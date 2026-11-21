"""The one shared ``normalize(text) -> list[str]`` used for indexing AND queries.

Pipeline: tokenise -> lowercase -> drop stop words -> Porter stem.
Technical tokens (tf-idf, 802.11, c++) are kept intact and, for hyphenated
compounds, their stemmed parts are emitted too so "tf idf" matches "tf-idf".
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nltk.stem.porter import PorterStemmer

from contextweave.textproc.tokenizer import is_technical, tokenize

_STOP_PATH = Path(__file__).with_name("stopwords_en.txt")
# NLTK's English stop-word list, bundled for reproducibility (no download needed).
STOPWORDS: frozenset[str] = frozenset(w.strip() for w in _STOP_PATH.read_text(encoding="utf-8").splitlines() if w.strip())
_STEMMER = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Porter-stem a single (non-technical) token, cached."""
    return _STEMMER.stem(token)


def normalize_token(tok: str, stopwords: frozenset[str] = STOPWORDS, use_stem: bool = True) -> list[str]:
    """Normalise one raw token into zero or more index terms."""
    if tok in stopwords or len(tok) < 2 and not tok.isdigit():
        return []
    if is_technical(tok):
        out = [tok]
        if "-" in tok:  # also index the parts of a hyphenated compound
            for part in tok.split("-"):
                if part and part not in stopwords and len(part) > 1:
                    out.append(stem(part) if use_stem and not is_technical(part) else part)
        return out
    return [stem(tok) if use_stem else tok]


def normalize(text: str, lowercase: bool = True, remove_stopwords: bool = True,
              use_stem: bool = True) -> list[str]:
    """Tokenise, lowercase, remove stop words and stem. Same function for docs and queries."""
    sw = STOPWORDS if remove_stopwords else frozenset()
    terms: list[str] = []
    for tok in tokenize(text, lowercase=lowercase):
        terms.extend(normalize_token(tok, sw, use_stem))
    return terms


def normalize_from_config(text: str, cfg) -> list[str]:
    """``normalize`` driven by the ``text`` section of the config."""
    t = cfg["text"]
    return normalize(text, lowercase=t.get("lowercase", True),
                     remove_stopwords=bool(t.get("stopwords")),
                     use_stem=t.get("stemmer") == "porter")
