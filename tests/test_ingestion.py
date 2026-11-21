from pathlib import Path

import pymupdf

from contextweave.ingestion.chunk import chunk_pages
from contextweave.ingestion.clean import clean_document, clean_page, drop_references, find_repeated_lines
from contextweave.ingestion.pdf_extract import extract_pages, split_into_docs


def _make_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)


def test_extract_and_clean_roundtrip(tmp_path):
    body = "Information retrieval finds material of an unstructured nature. " * 12
    pages = [f"COURSE NOTES HEADER\n{body}\nPage {i}" for i in range(1, 5)]
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, pages)
    raw = extract_pages(pdf)
    assert len(raw) == 4 and raw[0][0] == 1
    kept, skipped = clean_document(raw, min_page_chars=100)
    assert skipped == 0 and len(kept) == 4
    assert "COURSE NOTES HEADER" not in kept[0][1]
    assert split_into_docs(pdf, "T")[0].title == "T"


def test_hyphen_join_and_header_detection():
    pages = ["Header X\nretrie-\nval text\nfooter 1", "Header X\nmore text\nfooter 2", "Header X\nend\nfooter 3"]
    rep = find_repeated_lines(pages)
    assert "header x" in rep and "footer #" in rep
    assert clean_page(pages[0], rep) == "retrieval text"


def test_short_pages_skipped_and_references_dropped():
    pages = [(1, "a" * 500), (2, "b" * 500), (3, "tiny"), (4, "c" * 400 + "\nReferences\n[1] x")]
    kept, skipped = clean_document(pages, min_page_chars=300)
    assert skipped == 1 and all("[1] x" not in t for _, t in kept)
    assert len(drop_references(pages)) == 4


def test_chunk_sizes_and_overlap():
    words = " ".join(f"w{i}" for i in range(500))
    ps = chunk_pages([(1, words[:1000]), (2, words[1000:])], "doc0001", "T", "s", "IR", 200, 40)
    assert ps[0].passage_id == "doc0001_p0000" and ps[0].n_tokens == 200
    assert ps[1].text.split()[0] == "w160"  # stride 160 -> 40 words overlap
    assert sum(1 for p in ps if p.n_tokens < 40) == 0
    assert ps[-1].text.split()[-1] == "w499"
