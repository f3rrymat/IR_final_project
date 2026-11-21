"""Turn lexical explanations into rows the UI (or CLI) can display."""
from __future__ import annotations

from contextweave.core.types import ScoredPassage


def term_rows(result: ScoredPassage) -> list[dict]:
    """Per-term contribution rows sorted by contribution, plus each term's share of the score."""
    terms = (result.explanation or {}).get("terms", {})
    total = sum(v.get("score", 0.0) for v in terms.values()) or 1.0
    rows = [{"term": t, **v, "share": round(v.get("score", 0.0) / total, 3)} for t, v in terms.items()]
    return sorted(rows, key=lambda r: -r.get("score", 0.0))


def format_explanation(result: ScoredPassage) -> str:
    """One-line text explanation, e.g. for the CLI."""
    rows = term_rows(result)
    parts = [f"{r['term']}={r['score']:.2f}(tf={r['tf']})" for r in rows]
    elim = result.explanation.get("eliminated") if result.explanation else None
    s = " + ".join(parts) if parts else "(no lexical terms)"
    if elim:
        s += f"   [eliminated: {', '.join(elim)}]"
    return s
