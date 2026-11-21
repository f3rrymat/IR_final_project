"""Text processing: one shared normalize() for documents and queries."""
from contextweave.textproc.normalizer import normalize, normalize_from_config

__all__ = ["normalize", "normalize_from_config"]
