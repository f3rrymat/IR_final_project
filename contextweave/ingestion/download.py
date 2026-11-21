"""Download the open-licence source PDFs listed in ``data/sources.csv`` into ``data/raw/``.

The source list is committed; the PDFs are not. Every download (and every
failure) is logged to ``data/raw/download_log.csv`` so the corpus can be
reproduced and audited.

arXiv titles are resolved through the public arXiv API so the manifest shows
real paper titles instead of IDs.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger(__name__)

UA = "ContextWeave-IR-course-project/1.0 (academic, non-commercial)"


def _safe_name(url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])
    if not stem.lower().endswith(".pdf"):
        stem += ".pdf"
    return stem[-120:]


def ssl_context() -> ssl.SSLContext:
    """System CA store plus certifi's bundle.

    Some Python builds (e.g. the Microsoft Store Python on Windows) only trust the OS store, which can lack
    the intermediate/root a server needs -> CERTIFICATE_VERIFY_FAILED. Adding certifi fixes that.
    """
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:  # pragma: no cover
        pass
    return ctx


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:  # noqa: S310 (trusted list)
        return r.read()


def arxiv_titles(ids: list[str]) -> dict[str, str]:
    """Return {arxiv_id: title} using the arXiv export API (one batched call)."""
    if not ids:
        return {}
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids) + f"&max_results={len(ids)}"
    try:
        root = ET.fromstring(_fetch(url, timeout=60))
    except Exception as exc:  # pragma: no cover - network
        log.warning("arXiv title lookup failed: %s", exc)
        return {}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: dict[str, str] = {}
    for entry in root.findall("a:entry", ns):
        aid = entry.findtext("a:id", default="", namespaces=ns).rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        out[aid] = title
    return out


def download_all(sources_csv: Path, raw_dir: Path, sleep_s: float = 1.0) -> list[dict]:
    """Download every source; returns rows with local path, sha1 and status."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(sources_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["url"].rsplit("/", 1)[-1] for r in rows if "arxiv.org" in r["url"]]
    titles = arxiv_titles(ids)
    out = []
    for r in rows:
        dest = raw_dir / _safe_name(r["url"])
        status = "cached"
        if not dest.exists() or dest.stat().st_size == 0:
            try:
                data = _fetch(r["url"])
                if not data.startswith(b"%PDF"):
                    raise ValueError("not a PDF")
                dest.write_bytes(data)
                status = "downloaded"
                time.sleep(sleep_s)
            except Exception as exc:
                status = f"failed: {exc}"
                log.warning("download failed %s: %s", r["url"], exc)
        if "arxiv.org" in r["url"]:
            aid = r["url"].rsplit("/", 1)[-1]
            if aid in titles:
                r["title"] = titles[aid]
        sha = hashlib.sha1(dest.read_bytes()).hexdigest() if dest.exists() else ""
        out.append({**r, "file": dest.name, "sha1": sha, "status": status})
        log.info("%s %s", status, r["url"])
    with open(raw_dir / "download_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[2]
    res = download_all(root / "data" / "sources.csv", root / "data" / "raw")
    ok = sum(1 for r in res if not r["status"].startswith("failed"))
    print(f"{ok}/{len(res)} sources available")
