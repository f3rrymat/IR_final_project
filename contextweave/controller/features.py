"""Query construction and pre-retrieval query features (spec §8.5).

Features: number of terms, matched terms, share of unknown terms, mean and max
IDF, question word present, quotes present, Boolean operators present, and
formula-like tokens present.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from contextweave.core.types import Query

QUESTION_WORDS = {"how", "why", "what", "when", "where", "which", "who", "whom", "whose",
                  "can", "does", "do", "is", "are", "should", "could", "would", "explain"}
_FORMULA = re.compile(r"[=^∑√∫×÷|]|\b\w+\([^)]*\)|\b(?:log|exp|sqrt|sum|argmax|argmin)\b\s*[\w(]|\b[a-z]\d\b")
_BOOL = re.compile(r"\b(AND|OR|NOT)\b")


def query_features(raw: str, terms: list[str], idf: Callable[[str], float], df: Callable[[str], int]) -> dict[str, float]:
    """Compute the pre-retrieval features the router uses."""
    words = raw.lower().split()
    uniq = list(dict.fromkeys(terms))
    known = [t for t in uniq if df(t) > 0]
    idfs = [idf(t) for t in known] or [0.0]
    first = words[0].strip("?,.!") if words else ""
    return {
        "n_terms": float(len(uniq)),
        "n_words": float(len(words)),
        "matched_terms": float(len(known)),
        "unknown_share": (len(uniq) - len(known)) / len(uniq) if uniq else 1.0,
        "mean_idf": sum(idfs) / len(idfs),
        "max_idf": max(idfs),
        "question": float(first in QUESTION_WORDS or raw.strip().endswith("?")),
        "quotes": float('"' in raw),
        "boolean": float(bool(_BOOL.search(raw))),
        "formula": float(bool(_FORMULA.search(raw.lower()))),
    }


def build_query(raw: str, normalize: Callable[[str], list[str]], idf: Callable[[str], float],
                df: Callable[[str], int], filters: Optional[dict[str, Any]] = None) -> Query:
    """Normalise with the shared pipeline and attach features. Boolean operators are
    removed from the bag of terms (they are handled by the Boolean retriever)."""
    clean = _BOOL.sub(" ", raw).replace('"', " ")
    terms = normalize(clean)
    return Query(raw=raw, terms=terms, features=query_features(raw, terms, idf, df), filters=filters or {})
