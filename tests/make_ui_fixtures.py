"""Write the JSON fixtures the web UI's unit tests use, straight from the real API.

    python -m tests.make_ui_fixtures

Search responses come from the small fixture corpus (no model download); evaluation responses from the saved
results in experiments/results. tests/test_api.py::test_ui_fixtures_match_the_api fails when these files no longer
match what the API returns, so the UI tests can never drift from the backend.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "src" / "test" / "fixtures"

SEARCHES = {
    "search_auto": {"query": "semaphore sem_wait sem_post lock", "mode": "auto", "budget": 50, "tau": 0.99},
    "search_auto_stops": {"query": "tf-idf weight", "mode": "auto", "tau": 0.05},
    "search_skips": {"query": "semaphore lock", "mode": "auto", "budget": 1.5, "tau": 0.99},
    "search_bm25": {"query": "tf-idf weight", "mode": "bm25"},
}


def build() -> dict[str, dict]:
    from api.server import create_app
    from tests.conftest import make_fixture_system
    from contextweave.core import registry

    system = make_fixture_system()
    system.retrievers = {n: registry.get(n) for n in registry.available()}
    app = create_app(loader=lambda c: system, warm=False, is_built=lambda c: c in ("cs", "scifact"))
    out: dict[str, dict] = {}
    with TestClient(app) as c:
        out["collections"] = c.get("/api/collections").json()
        for name, body in SEARCHES.items():
            r = c.post("/api/search", json={"collection": "cs", **body})
            r.raise_for_status()
            out[name] = r.json()
        out["evaluation_scifact"] = c.get("/api/evaluation/scifact").json()
        out["evaluation_cs"] = c.get("/api/evaluation/cs").json()
    for name in list(out):  # latency is machine-dependent: pin it so fixtures are reproducible
        _pin_latency(out[name])
    return out


def _pin_latency(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("latency_ms", "encode_ms") and isinstance(v, (int, float)):
                x[k] = 1.0
            else:
                _pin_latency(v)
    elif isinstance(x, list):
        for v in x:
            _pin_latency(v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in build().items():
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                          newline="")
        print("wrote", OUT / f"{name}.json")


if __name__ == "__main__":
    main()
