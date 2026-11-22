"""ContextWeave web API + UI server.

    python main.py serve            # http://127.0.0.1:8000  (UI built into web/dist)
    uvicorn api.server:app --reload # API only, for development next to `npm run dev` in web/

Endpoints (all JSON):
    GET  /api/health
    GET  /api/collections                         which collections are built, their tuned defaults and costs
    POST /api/search                              one query through the controller (or one fixed method)
    GET  /api/collections/{c}/passages/{pid}      full text of a passage
    GET  /api/evaluation/{c}?split=test           saved experiment results (never recomputed here)
    GET  /api/judge/{c}                           judging progress for the cs benchmark
    GET  /api/judge/{c}/item?qid=                 next passage to judge
    POST /api/judge/{c}                           save one grade (0/1/2)
    POST /api/judge/{c}/pool                      build the judging pool
"""
from __future__ import annotations

import csv
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Query as Q
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.serialize import evaluation_payload, search_payload
from contextweave.controller.engine import models_dir, tuned_defaults
from contextweave.core.config import ROOT, load_config
from contextweave.evaluation import benchmark as bench

log = logging.getLogger(__name__)

WEB_DIST = ROOT / "web" / "dist"
RESULTS = ROOT / "experiments" / "results"
COLLECTIONS = {
    "cs": {"label": "CS course material", "short": "CS",
           "description": "163 open textbooks, lecture notes and papers across IR, OS, Networks, DBMS and Algorithms",
           "examples": ["How can a search engine identify the importance of a webpage?", "TF-IDF formula and inverse document frequency",
                        "Compare BM25 and TF-IDF", "semaphore AND NOT deadlock", "TCP Reno fast retransmit fast recovery"]},
    "scifact": {"label": "SciFact", "short": "SciFact",
                "description": "BEIR SciFact - scientific claims checked against research abstracts",
                "examples": ["0-dimensional biomaterials show inductive properties.",
                             "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
                             "Statins reduce the risk of heart attacks"]},
    "nfcorpus": {"label": "NFCorpus", "short": "NFCorpus",
                 "description": "BEIR NFCorpus - nutrition questions answered by medical abstracts",
                 "examples": ["Why Deep Fried Foods May Cause Cancer", "statins", "breast cancer", "vitamin D"]},
}
MODES = ["auto", "bm25", "tfidf", "dense", "hybrid", "boolean", "bm25_champion", "route"]
LADDER = ("bm25_champion", "bm25", "dense", "expansion")
NOVEL = {"qpp_signals": None, "voc_policy": "voc_gain_model.json", "risk_controlled_tau": "risk_tau.json",
         "qpp_fusion": "qpp_fusion.json"}


class SearchRequest(BaseModel):
    collection: str
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["auto", "bm25", "tfidf", "dense", "hybrid", "boolean", "bm25_champion", "route"] = "auto"
    budget: Optional[float] = Field(default=None, gt=0, le=1000)
    tau: Optional[float] = Field(default=None, ge=0, le=1)
    expansion: Optional[bool] = None
    features: dict[str, bool] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=50)


class Judgement(BaseModel):
    qid: str
    passage_id: str
    grade: int = Field(ge=0, le=2)
    annotator: str = Field(default="", max_length=60)


class Systems:
    """One loaded System per collection, loaded on first use, plus a lock per system.

    Retrievers keep per-call statistics on themselves, so searches on the same system are serialised.
    """

    def __init__(self, loader: Callable[[str], object]):
        self._loader, self._systems, self._locks = loader, {}, {}
        self._guard = threading.Lock()

    def lock(self, c: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(c, threading.Lock())

    def get(self, c: str):
        with self.lock(c):
            if c not in self._systems:
                self._systems[c] = self._loader(c)
            return self._systems[c]

    def loaded(self) -> list[str]:
        return sorted(self._systems)


def default_loader(collection: str):
    from contextweave.system import load_system

    cfg = load_config(overrides={"collection": collection})
    system = load_system(cfg, collection=collection, build_missing=False)
    if system.embedder is not None:  # load the model now so the first search is not slow
        system.embedder.encode(["warm-up"])
    return system


def _is_built(c: str, cfg) -> bool:
    return (Path(cfg["paths"]["indexes"]) / c / "inverted.pkl").exists() and \
        (Path(cfg["paths"]["processed"]) / c / "passages.jsonl").exists()


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def _domains(c: str) -> list[str]:
    if c != "cs":
        return []
    p = ROOT / "data" / "manifest.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return sorted({r["domain"] for r in csv.DictReader(f) if r.get("domain")})


def create_app(loader: Callable[[str], object] = default_loader, cfg=None, warm: bool = True,
               is_built: Optional[Callable[[str], bool]] = None) -> FastAPI:
    """``loader`` and ``is_built`` are injectable so tests can serve the small fixture corpus."""
    cfg = cfg or load_config()
    ready = is_built or (lambda c: _is_built(c, cfg))
    systems = Systems(loader)
    counts: dict[str, int] = {}

    def built() -> list[str]:
        return [c for c in COLLECTIONS if ready(c)]

    def warm_all(names: list[str]) -> None:
        """Load every built collection (indexes + embedding model) in the background, one after another, so
        no search has to wait for it. A search on a collection still loading simply waits for that one."""
        import time

        for c in names:
            t = time.perf_counter()
            try:
                systems.get(c)
                print(f"ContextWeave: {c} ready ({time.perf_counter() - t:.1f}s)", flush=True)
            except Exception as e:  # keep serving; the search endpoint will report the error for this collection
                print(f"ContextWeave: could not load {c}: {e}", flush=True)
        print("ContextWeave: all collections loaded, searches are now instant", flush=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        b = built()
        if warm and b:
            print(f"ContextWeave: loading {', '.join(b)} in the background (about 15 s on first start)", flush=True)
            threading.Thread(target=warm_all, args=(b,), daemon=True).start()
        yield

    app = FastAPI(title="ContextWeave", version="2.0", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None,
                  lifespan=lifespan)
    app.state.systems = systems

    def need(c: str):
        if c not in COLLECTIONS:
            raise HTTPException(404, f"unknown collection {c!r}")
        if not ready(c):
            raise HTTPException(409, f"collection {c!r} is not built yet - run: python main.py build-index --collection {c}")
        return systems.get(c)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "built": built(), "loaded": systems.loaded(), "ui_built": WEB_DIST.exists()}

    @app.get("/api/collections")
    def collections():
        out = []
        for c, meta in COLLECTIONS.items():
            ok = ready(c)
            costs = {**cfg["costs"], **(cfg.get("costs_by_collection") or {}).get(c, {})}
            if ok and c not in counts:
                pf = Path(cfg["paths"]["processed"]) / c / "passages.jsonl"
                counts[c] = _count_lines(pf) if pf.exists() else None
            out.append({
                "id": c, **meta, "ready": ok, "passages": counts.get(c),
                "domains": _domains(c), "defaults": tuned_defaults(c, cfg),
                "expansion_default": bool(cfg["expansion"]["enabled"]),
                "costs": {m: round(float(costs[m]), 3) for m in costs},
                "max_budget": float(round(sum(costs[m] for m in LADDER) + 1)),
                "features": {f: (fn is None or (models_dir(c) / fn).exists()) for f, fn in NOVEL.items()},
                "has_results": any(RESULTS.glob(f"eval_{c}_*_latest.json")),
                "judging": c == "cs",
            })
        return {"collections": out, "modes": MODES, "default": (built() or ["cs"])[0]}

    @app.post("/api/search")
    def search(req: SearchRequest):
        system = need(req.collection)
        q = req.query.strip()
        if not q:
            raise HTTPException(422, "empty query")
        d = tuned_defaults(req.collection, cfg)
        feats = {f: bool(req.features.get(f, False)) for f in NOVEL}
        tau = None if feats["risk_controlled_tau"] else (req.tau if req.tau is not None else d["tau"])
        with systems.lock(req.collection):
            eng = system.engine(features=feats, expansion=req.expansion, tau=tau,
                                budget=req.budget if req.budget is not None else d["budget"])
            resp = eng.search(q, mode=req.mode, k=req.k,
                              filters={"domains": req.domains} if req.domains else {})
            return search_payload(resp, system)

    @app.get("/api/collections/{c}/passages/{pid}")
    def passage(c: str, pid: str):
        system = need(c)
        p = system.store.get(pid)
        if p is None:
            raise HTTPException(404, f"no passage {pid!r} in {c}")
        return {**p.to_dict()}

    @app.get("/api/evaluation/{c}")
    def evaluation(c: str, split: Optional[str] = Q(default=None)):
        if c not in COLLECTIONS:
            raise HTTPException(404, f"unknown collection {c!r}")
        files = {p.name.split("_")[2]: p for p in sorted(RESULTS.glob(f"eval_{c}_*_latest.json"))}
        if not files:
            return {"available": [], "result": None}
        s = split if split in files else ("test" if "test" in files else sorted(files)[0])
        d = json.loads(files[s].read_text(encoding="utf-8"))
        return {"available": sorted(files), "result": evaluation_payload(d)}

    # ------------------------------------------------------------------ judging (cs benchmark)
    def pool_path(c: str) -> Path:
        return bench.bench_dir(c) / "pool.json"

    def judge_state(c: str) -> dict:
        if c not in COLLECTIONS:
            raise HTTPException(404, f"unknown collection {c!r}")
        pp = pool_path(c)
        if not pp.exists():
            return {"pool_ready": False, "done": 0, "total": 0, "queries": []}
        pool = json.loads(pp.read_text(encoding="utf-8"))
        qrels = bench.load_qrels(c)
        queries = {q.qid: q for q in bench.load_queries(c)}
        rows = [{"qid": qid, "text": queries[qid].text if qid in queries else qid,
                 "type": queries[qid].type if qid in queries else "", "domain": queries[qid].domain if qid in queries else "",
                 "judged": sum(p in qrels.get(qid, {}) for p in pids), "total": len(pids)}
                for qid, pids in pool.items()]
        return {"pool_ready": True, "done": sum(r["judged"] for r in rows), "total": sum(r["total"] for r in rows),
                "queries": rows, "pool": pool, "qrels": qrels, "qmap": queries}

    def public(state: dict) -> dict:
        return {k: v for k, v in state.items() if k not in ("pool", "qrels", "qmap")}

    @app.get("/api/judge/{c}")
    def judge(c: str):
        return public(judge_state(c))

    @app.get("/api/judge/{c}/item")
    def judge_item(c: str, qid: Optional[str] = None, passage_id: Optional[str] = None):
        st = judge_state(c)
        if not st["pool_ready"]:
            raise HTTPException(409, "no judging pool yet")
        pool, qrels = st["pool"], st["qrels"]
        if qid is not None and qid not in pool:
            raise HTTPException(404, f"query {qid!r} is not in the pool")
        if qid and passage_id:
            items = [(qid, passage_id)] if passage_id in pool[qid] else []
        else:
            cand = [(q, p) for q in ([qid] if qid else pool) for p in pool[q]]
            items = [x for x in cand if x[1] not in qrels.get(x[0], {})] or (cand if qid else [])
        if not items:
            return {"item": None, **public(st)}
        q, pid = items[0]
        system = need(c)
        p = system.store[pid]
        bq = st["qmap"].get(q)
        return {"item": {"qid": q, "query": bq.text if bq else q, "type": bq.type if bq else "",
                         "domain": bq.domain if bq else "", "passage": p.to_dict(),
                         "grade": qrels.get(q, {}).get(pid), "position": pool[q].index(pid) + 1,
                         "of": len(pool[q])}, **public(st)}

    @app.post("/api/judge/{c}")
    def judge_save(c: str, j: Judgement):
        st = judge_state(c)
        if not st["pool_ready"] or j.qid not in st["pool"] or j.passage_id not in st["pool"][j.qid]:
            raise HTTPException(404, "that (query, passage) pair is not in the judging pool")
        bench.write_qrel(c, j.qid, j.passage_id, j.grade, j.annotator.strip().lower().replace(" ", "_"))
        return public(judge_state(c))

    @app.post("/api/judge/{c}/pool")
    def judge_pool(c: str):
        system = need(c)
        with systems.lock(c):
            pool = bench.build_pool(system.engine(), bench.load_queries(c))
        pool_path(c).write_text(json.dumps(pool, indent=1), encoding="utf-8", newline="")
        return public(judge_state(c))

    # ------------------------------------------------------------------ the React UI
    if (WEB_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "unknown API route")
        f = WEB_DIST / path
        if path and f.is_file() and WEB_DIST in f.resolve().parents:
            return FileResponse(f)
        if (WEB_DIST / "index.html").exists():
            return FileResponse(WEB_DIST / "index.html")
        return HTMLResponse("<h1>ContextWeave API is running</h1><p>The UI is not built yet: "
                            "<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, then reload. "
                            "API docs: <a href='/api/docs'>/api/docs</a></p>")

    return app


app = create_app()
