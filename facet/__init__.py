"""FACET retrieval core. No data, model loading, GPU allocation or background jobs on import."""
from .chunking import make_blocks
from .context import make_context_text
from .retrieval import pack_blocks
from .scoring import score_entries, fit_thresholds
from .selection import select_entries
from .pipeline import build_index, retrieve

__all__ = ['make_blocks', 'make_context_text', 'pack_blocks', 'score_entries',
           'fit_thresholds', 'select_entries', 'build_index', 'retrieve']
