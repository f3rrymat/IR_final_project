"""PassageStore: id <-> integer index <-> Passage lookup shared by all packages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from contextweave.core.types import Passage


class PassageStore:
    """Holds every passage of a collection in index order."""

    def __init__(self, passages: Iterable[Passage]):
        self.passages: list[Passage] = list(passages)
        self.index_of: dict[str, int] = {p.passage_id: i for i, p in enumerate(self.passages)}
        if len(self.index_of) != len(self.passages):
            raise ValueError("duplicate passage ids")
        self.doc_of: list[str] = [p.doc_id for p in self.passages]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PassageStore":
        with open(path, encoding="utf-8") as f:
            return cls(Passage(**json.loads(line)) for line in f if line.strip())

    def __len__(self) -> int:
        return len(self.passages)

    def __iter__(self) -> Iterator[Passage]:
        return iter(self.passages)

    def __getitem__(self, pid: str) -> Passage:
        return self.passages[self.index_of[pid]]

    def get(self, pid: str) -> Passage | None:
        i = self.index_of.get(pid)
        return None if i is None else self.passages[i]

    def domains(self) -> list[str]:
        return sorted({p.domain for p in self.passages})

    def ids_in_domains(self, domains: Iterable[str]) -> set[str]:
        ds = set(domains)
        return {p.passage_id for p in self.passages if p.domain in ds}
