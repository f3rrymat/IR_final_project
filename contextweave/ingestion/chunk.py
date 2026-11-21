"""Split cleaned pages into overlapping passages of about ``max_tokens`` words."""
from __future__ import annotations

from contextweave.core.types import Passage


def chunk_pages(pages: list[tuple[int, str]], doc_id: str, title: str, source: str, domain: str,
                max_tokens: int = 200, overlap: int = 40) -> list[Passage]:
    """Sliding window over the document's words (stride = max_tokens - overlap).

    A passage's ``page`` is the page of its first word. A trailing window shorter
    than ``overlap`` words is merged into the previous passage's span instead of
    being emitted alone.
    """
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    words: list[str] = []
    word_page: list[int] = []
    for pno, text in pages:
        w = text.split()
        words.extend(w)
        word_page.extend([pno] * len(w))
    stride = max_tokens - overlap
    passages: list[Passage] = []
    start, i = 0, 0
    n = len(words)
    while start < n:
        end = min(start + max_tokens, n)
        if n - end < overlap and end < n:  # avoid a tiny tail passage
            end = n
        piece = words[start:end]
        passages.append(Passage(
            passage_id=f"{doc_id}_p{i:04d}", doc_id=doc_id, text=" ".join(piece), title=title,
            source=source, page=word_page[start] if word_page else None, domain=domain,
            n_tokens=len(piece)))
        i += 1
        if end >= n:
            break
        start += stride
    return passages
