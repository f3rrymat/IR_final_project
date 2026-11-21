"""Page-by-page PDF text extraction with PyMuPDF.

A single large PDF (an open textbook) can be split into several logical
documents, one per chapter, using either the PDF outline ("toc") or chapter
heading pages ("heading"). Each logical document keeps its page numbers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)


@dataclass
class RawDoc:
    """A logical document: title plus a list of (page_number, text)."""

    title: str
    source: str
    pages: list[tuple[int, str]] = field(default_factory=list)


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(1-based page number, text)] for every page of a PDF."""
    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]


def _toc_chapters(path: Path) -> list[tuple[str, int]]:
    with pymupdf.open(path) as doc:
        return [(title.strip(), page) for level, title, page in doc.get_toc() if level == 1]


_HEAD_NUM = re.compile(r"^\s*(\d{1,2})\s*$")


def _heading_chapters(pages: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Detect chapter start pages: first line is a bare chapter number N and the
    next line is a title (not "N Title" running header). Numbers must increase."""
    out, expected = [], 1
    for pno, text in pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        m = _HEAD_NUM.match(lines[0])
        if m and int(m.group(1)) == expected and not lines[1][:1].isdigit():
            join_next = len(lines[1]) < 35 and not lines[2][:1] in "\"“‘'[(" and len(lines[2]) < 35
            title = f"{lines[1]} {lines[2]}" if join_next else lines[1]
            out.append((f"{expected} {title}", pno))
            expected += 1
    return out


def split_into_docs(path: Path, base_title: str, mode: str = "none",
                    exclude: str | None = None) -> list[RawDoc]:
    """Extract a PDF and split it into chapter documents.

    mode: "none" (whole file is one document), "toc" (PDF outline, level 1),
    or "heading" (chapter heading pages). ``exclude`` is a regex of chapter
    titles to drop (front matter, index, solutions...).
    """
    pages = extract_pages(path)
    if mode == "none":
        return [RawDoc(base_title, path.name, pages)]
    chapters = _toc_chapters(path) if mode == "toc" else _heading_chapters(pages)
    if not chapters:
        log.warning("no chapters found in %s; keeping it as one document", path.name)
        return [RawDoc(base_title, path.name, pages)]
    ex = re.compile(exclude, re.I) if exclude else None
    docs = []
    for i, (title, start) in enumerate(chapters):
        end = chapters[i + 1][1] - 1 if i + 1 < len(chapters) else len(pages)
        if ex and ex.search(title):
            continue
        title_clean = re.sub(r"\s+", " ", title)
        docs.append(RawDoc(f"{base_title} — {title_clean}", path.name, pages[start - 1:end]))
    return docs
