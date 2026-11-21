"""Config loader: all tunable values live in ``config/settings.yaml``."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "settings.yaml"


class Config(dict):
    """A dict with attribute access for nested sections (``cfg.bm25.k1``)."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def flag(self, name: str, default: bool = False) -> bool:
        """Read an on/off flag from the ``features`` section."""
        return bool((self.get("features") or {}).get(name, default))


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load YAML config, apply optional overrides, and resolve paths relative to repo root."""
    p = Path(path) if path else DEFAULT_PATH
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = deep_update(data, overrides or {})
    paths = data.setdefault("paths", {})
    for key, val in list(paths.items()):
        if isinstance(val, str) and not Path(val).is_absolute():
            paths[key] = str(ROOT / val)
    return Config(data)


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch optional at import time
        pass
