"""Query-biased snippets: the window of a passage containing the most query terms,
with the matching words highlighted."""
from __future__ import annotations

import html
from typing import Callable, Iterable


def _best_window(words: list[str], qset: set[str], normalize: Callable[[str], list[str]],
                 window: int) -> tuple[list[int], int, int, int]:
    """(hit flags, start, width, hits) of the ``window``-word span with the most query-term words."""
    hit = [1 if qset.intersection(normalize(w)) else 0 for w in words]
    w = min(window, len(words))
    cur = sum(hit[:w])
    best, best_start = cur, 0
    for s in range(1, len(words) - w + 1):  # sliding window, O(n)
        cur += hit[s + w - 1] - hit[s - 1]
        if cur > best:
            best, best_start = cur, s
    return hit, best_start, w, best


def make_snippet(text: str, query_terms: Iterable[str], normalize: Callable[[str], list[str]],
                 window: int = 40, mark: tuple[str, str] = ("<mark>", "</mark>")) -> dict:
    """Return {"html": highlighted window, "start": word offset, "hits": n matched words}.

    The window size is in words. Each word is normalised with the shared pipeline so
    "Retrieval" highlights for the query term "retriev".
    """
    words = text.split()
    if not words:
        return {"html": "", "start": 0, "hits": 0}
    hit, start, w, best = _best_window(words, set(query_terms), normalize, window)
    out = []
    for i in range(start, start + w):
        token = html.escape(words[i])
        out.append(f"{mark[0]}{token}{mark[1]}" if hit[i] else token)
    prefix = "… " if start > 0 else ""
    suffix = " …" if start + w < len(words) else ""
    return {"html": prefix + " ".join(out) + suffix, "start": start, "hits": best}


def snippet_segments(text: str, query_terms: Iterable[str], normalize: Callable[[str], list[str]],
                     window: int = 40) -> dict:
    """Same window as :func:`make_snippet`, as plain-text segments for a client to render safely.

    Returns {"segments": [[text, is_match], ...], "before": bool, "after": bool, "hits": int}; adjacent words
    with the same flag are merged, so a client never has to inject HTML.
    """
    words = text.split()
    if not words:
        return {"segments": [], "before": False, "after": False, "hits": 0}
    hit, start, w, best = _best_window(words, set(query_terms), normalize, window)
    segs: list[list] = []
    for i in range(start, start + w):
        flag = bool(hit[i])
        if segs and segs[-1][1] == flag:
            segs[-1][0] += " " + words[i]
        elif not segs:
            segs.append([words[i], flag])
        elif flag:  # keep the separating space outside the highlight
            segs[-1][0] += " "
            segs.append([words[i], flag])
        else:
            segs.append([" " + words[i], flag])
    return {"segments": segs, "before": start > 0, "after": start + w < len(words), "hits": best}
