"""Tokenizer plugins."""

from __future__ import annotations

# Import all tokenizer plugins to trigger registration
from . import icu
from . import spacy
from . import sentencepiece
from . import huggingface

__all__ = ["icu", "spacy", "sentencepiece", "huggingface"]
