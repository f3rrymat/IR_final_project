"""Tokeniser that keeps technical tokens intact.

Examples kept as single tokens: ``tf-idf``, ``802.11``, ``k-means``, ``c++``,
``b+tree``, ``o(n log n)`` -> ``o``, ``n``, ``log``, ``n``.
Hyphenated compounds are also split into their parts by the normaliser so that
"tf idf" and "tf-idf" still match.
"""
from __future__ import annotations

import re

# Order matters: the most specific patterns first.
_TOKEN_RE = re.compile(
    r"""
    \d+(?:\.\d+)+                 # version / standard numbers: 802.11, 3.14, 1.2.3
  | [a-z][a-z0-9]*\+\+            # c++
  | [a-z0-9]+(?:\+[a-z0-9]+)+     # b+tree
  | [a-z0-9]+(?:-[a-z0-9]+)+      # tf-idf, k-means, write-ahead
  | [a-z0-9]+(?:'[a-z]+)?         # words, numbers, contractions (don't)
    """,
    re.VERBOSE,
)


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into tokens, preserving technical tokens."""
    if lowercase:
        text = text.lower()
    # unify typographic dashes/quotes so they behave like ASCII
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-")
    text = text.replace("’", "'").replace("‘", "'")
    return _TOKEN_RE.findall(text)


def is_technical(token: str) -> bool:
    """True for tokens that must not be stemmed (contain digits or symbols)."""
    return any(c in token for c in "-.+") or any(c.isdigit() for c in token)
