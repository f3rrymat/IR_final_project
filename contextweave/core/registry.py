"""Name -> Retriever registry.

The controller and the UI look retrievers up by name, so new methods can be
added (e.g. by a team member on their branch) without editing other modules.
"""
from __future__ import annotations

import logging

from contextweave.core.retriever import Retriever

log = logging.getLogger(__name__)
_REGISTRY: dict[str, Retriever] = {}


def register(retriever: Retriever) -> None:
    """Register (or replace) a retriever under ``retriever.name``."""
    if not isinstance(retriever, Retriever):
        raise TypeError(f"{retriever!r} does not implement the Retriever protocol")
    if retriever.name in _REGISTRY:
        log.debug("replacing retriever %s", retriever.name)
    _REGISTRY[retriever.name] = retriever


def get(name: str) -> Retriever:
    """Return the retriever registered as ``name``; raises KeyError with the options."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown retriever {name!r}; available: {available()}") from None


def available() -> list[str]:
    """Sorted list of registered retriever names."""
    return sorted(_REGISTRY)


def clear() -> None:
    """Remove all registrations (used by tests)."""
    _REGISTRY.clear()
