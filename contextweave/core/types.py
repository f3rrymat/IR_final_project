"""Shared data types (spec section 6).

Every package exchanges data only through these types. Keep them stable;
adding optional fields with defaults is allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Passage:
    """One retrievable unit of text (about 200 tokens)."""

    passage_id: str  # e.g. "doc0042_p0013"
    doc_id: str
    text: str
    title: str
    source: str  # file name or URL
    page: Optional[int]
    domain: str  # "IR" | "OS" | "Networks" | "DBMS" | "Algorithms" (or collection name)
    n_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Query:
    """A user query after normalisation."""

    raw: str
    terms: list[str]  # normalised terms
    features: dict[str, float] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    # Optional: extra weighted terms (used by query expansion). term -> weight.
    term_weights: dict[str, float] = field(default_factory=dict)
    # Optional per-query artefacts computed once and reused (e.g. the query embedding).
    cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass
class ScoredPassage:
    """A passage with a score from one retriever (or from fusion)."""

    passage_id: str
    doc_id: str
    score: float
    method: str  # name of the retriever that produced it
    rank: int = 0
    norm_score: Optional[float] = None
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """One logged decision of the adaptive controller."""

    action: str  # "route" | "retrieve" | "confidence" | "escalate" | "stop"
    method: Optional[str]
    cost: float
    budget_left: float
    confidence: Optional[float]
    signals: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    note: str = ""


@dataclass
class SearchResponse:
    """Everything the UI and evaluation need for one query."""

    query: Query
    results: list[ScoredPassage]
    trace: list[TraceStep]
    total_cost: float
    total_latency_ms: float
    mode: str  # "auto" | "bm25" | "tfidf" | "dense" | "hybrid" | "boolean"
    extras: dict[str, Any] = field(default_factory=dict)
