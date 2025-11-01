"""Feature extractor plugins."""

from __future__ import annotations

# Import all feature extractor plugins to trigger registration
from . import bow
from . import tfidf
from . import fasttext
from . import sbert

__all__ = ["bow", "tfidf", "fasttext", "sbert"]
