"""Ablations only: remove one part of S8 at a time (expansion, routing, escalation,
pruning) plus the tau x budget sweep, on the test split. `run_all.py` already
includes these; this script re-runs just this part.

    python experiments/ablations.py scifact nfcorpus
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_all import run_systems  # noqa: E402

if __name__ == "__main__":
    for c in sys.argv[1:] or ["cs"]:
        print(run_systems(None, c, "test", systems=["S5", "S8"], latency=False, ablations=True, sweep=True))
