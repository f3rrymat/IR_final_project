"""Percentile-calibrated QPP fusion weights — novel extension (flag ``qpp_fusion``).

Raw BM25 and cosine scores live on different scales, so a list's "how sure am I"
signal cannot be compared directly across methods. We compute one scale-free
predictor per list, the relative top gap

    rg(list) = (s_1 - mean(s_1..s_10)) / |mean(s_1..s_10)|

and convert it to a percentile against the same method's values on the
validation queries. The lexical weight then follows the two lists' relative
standing:

    alpha_qpp(q) = pct_L / (pct_L + pct_V)
    alpha(q)     = mix * alpha_rule(q) + (1 - mix) * alpha_qpp(q)
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

from contextweave.core.types import ScoredPassage


def relative_gap(results: list[ScoredPassage]) -> float:
    s = [r.score for r in results[:10]]
    if len(s) < 2:
        return 0.0
    mean = sum(s) / len(s)
    return (s[0] - mean) / (abs(mean) + 1e-9)


class QppFusion:
    def __init__(self, reference: dict[str, list[float]], mix: float = 0.5):
        self.ref = {k: sorted(v) for k, v in reference.items()}
        self.mix = mix

    def percentile(self, method: str, value: float) -> float:
        ref = self.ref.get(method) or self.ref.get("bm25") or [0.0]
        return (bisect.bisect_left(ref, value) + 0.5) / (len(ref) + 1)

    def alpha(self, rule_alpha: float, lex_method: str, lex: list[ScoredPassage], dense: list[ScoredPassage]) -> tuple[float, dict]:
        pl = self.percentile(lex_method, relative_gap(lex))
        pv = self.percentile("dense", relative_gap(dense))
        a_qpp = pl / (pl + pv)
        a = self.mix * rule_alpha + (1 - self.mix) * a_qpp
        return a, {"pct_lexical": round(pl, 3), "pct_dense": round(pv, 3), "alpha_qpp": round(a_qpp, 3),
                   "alpha_rule": rule_alpha}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"reference": self.ref, "mix": self.mix}), encoding="utf-8", newline="")

    @classmethod
    def load(cls, path: str | Path) -> "QppFusion":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["reference"], d["mix"])
