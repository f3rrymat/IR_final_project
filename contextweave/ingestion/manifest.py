"""Corpus pipeline: sources -> extract -> clean -> chunk -> passages.jsonl + manifest.csv.

Outputs
  data/manifest.csv                       one row per logical document (committed)
  data/processed/<collection>/passages.jsonl
  data/processed/<collection>/rejected.csv  files / chapters that were rejected and why
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import pymupdf

from contextweave.core.config import ROOT, Config
from contextweave.core.types import Passage
from contextweave.ingestion.chunk import chunk_pages
from contextweave.ingestion.clean import clean_document
from contextweave.ingestion.download import _safe_name
from contextweave.ingestion.pdf_extract import split_into_docs

log = logging.getLogger(__name__)

MANIFEST_FIELDS = ["doc_id", "title", "domain", "source_file", "url", "license", "pages_kept",
                   "pages_skipped", "n_passages", "n_words"]


def _pdf_title(path: Path, fallback: str) -> str:
    """Best-effort human title: PDF metadata, else the first substantial line of page 1."""
    try:
        with pymupdf.open(path) as d:
            meta = (d.metadata or {}).get("title") or ""
            if 5 <= len(meta) <= 120 and not meta.lower().endswith((".dvi", ".pdf", ".tex")):
                return meta.strip()
            first = d[0].get_text("text") if d.page_count else ""
    except Exception:  # pragma: no cover
        return fallback
    for line in first.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 90 and re.search(r"[A-Za-z]{4}", line) and not re.match(r"^\d", line):
            return line
    return fallback


def _source_titles(raw_dir: Path) -> dict[str, str]:
    """Titles resolved during download (arXiv API), keyed by URL."""
    p = raw_dir / "download_log.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["url"]: r["title"] for r in csv.DictReader(f)}


def write_passages(passages: Iterable[Passage], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for p in passages:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_passages(path: Path) -> list[Passage]:
    """Read passages.jsonl back into Passage objects."""
    with open(path, encoding="utf-8") as f:
        return [Passage(**json.loads(line)) for line in f if line.strip()]


def build_corpus(cfg: Config, sources_csv: Path | None = None, raw_dir: Path | None = None,
                 out_dir: Path | None = None, manifest_path: Path | None = None) -> dict:
    """Run the whole ingestion pipeline. Returns corpus statistics."""
    sources_csv = sources_csv or ROOT / "data" / "sources.csv"
    raw_dir = raw_dir or Path(cfg["paths"]["raw"])
    out_dir = out_dir or Path(cfg["paths"]["processed"]) / cfg["collection"]
    manifest_path = manifest_path or ROOT / "data" / "manifest.csv"
    ing, ch = cfg["ingestion"], cfg["chunking"]
    titles = _source_titles(raw_dir)

    with open(sources_csv, newline="", encoding="utf-8") as f:
        sources = list(csv.DictReader(f))
    manifest, rejected, passages = [], [], []
    doc_no = 0
    for src in sources:
        path = raw_dir / _safe_name(src["url"])
        if not path.exists():
            rejected.append({"source": src["url"], "reason": "file missing (download failed)"})
            continue
        # arXiv titles were resolved by the downloader; other titles are written in sources.csv
        base = titles.get(src["url"], src["title"]) if "arxiv.org" in src["url"] else src["title"]
        try:
            raw_docs = split_into_docs(path, base, src.get("split_mode", "none"), src.get("exclude") or None)
            if src.get("titles"):  # chapter titles written in sources.csv override detected ones
                names = [t.strip() for t in src["titles"].split("|")]
                for rd, name in zip(raw_docs, names):
                    rd.title = f"{base} — {name}"
        except Exception as exc:
            rejected.append({"source": path.name, "reason": f"extraction error: {exc}"})
            continue
        for rd in raw_docs:
            pages, skipped = clean_document(rd.pages, ing["min_page_chars"], ing["header_footer_repeat"],
                                            bool(ing.get("split_glued_words", False)))
            n_chars = sum(len(t) for _, t in pages)
            if n_chars < ing["min_doc_chars"]:
                rejected.append({"source": f"{path.name} :: {rd.title}", "reason": f"too little text ({n_chars} chars)"})
                continue
            doc_no += 1
            doc_id = f"doc{doc_no:04d}"
            ps = chunk_pages(pages, doc_id, rd.title, path.name, src["domain"], ch["max_tokens"], ch["overlap"])
            passages.extend(ps)
            manifest.append({"doc_id": doc_id, "title": rd.title, "domain": src["domain"],
                             "source_file": path.name, "url": src["url"], "license": src["license"],
                             "pages_kept": len(pages), "pages_skipped": skipped, "n_passages": len(ps),
                             "n_words": sum(p.n_tokens for p in ps)})
            log.info("%s %-70s pages=%d passages=%d", doc_id, rd.title[:70], len(pages), len(ps))

    if not passages:  # never overwrite the committed manifest with an empty one
        raise FileNotFoundError(f"no source PDFs found in {raw_dir} - run `python -m contextweave.ingestion.download` "
                                f"first ({len(rejected)} sources missing or rejected)")
    write_passages(passages, out_dir / "passages.jsonl")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)
    with open(out_dir / "rejected.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "reason"])
        w.writeheader()
        w.writerows(rejected)
    for r in rejected:
        log.warning("rejected %s: %s", r["source"], r["reason"])
    return corpus_stats(passages, manifest, rejected)


def corpus_stats(passages: list[Passage], manifest: list[dict], rejected: list[dict]) -> dict:
    by_domain: dict[str, dict[str, int]] = {}
    for m in manifest:
        d = by_domain.setdefault(m["domain"], {"documents": 0, "passages": 0})
        d["documents"] += 1
        d["passages"] += m["n_passages"]
    lens = [p.n_tokens for p in passages] or [0]
    return {"documents": len(manifest), "passages": len(passages), "rejected": len(rejected),
            "mean_tokens": round(mean(lens), 1), "median_tokens": median(lens), "by_domain": by_domain}
