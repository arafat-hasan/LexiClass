#!/usr/bin/env python3
"""
Compatibility layer: re-exports dataset helpers from the `lexiclass.datasets` package.
Prefer importing from `lexiclass.datasets.wikipedia` directly.
"""

from lexiclass.datasets.wikipedia import (
    clean_wikipedia_text,
    categorize_wikipedia_article,
    is_valid_article,
    load_wikipedia_dataset,
    iter_wikipedia_dataset,
)

__all__ = [
    "clean_wikipedia_text",
    "categorize_wikipedia_article",
    "is_valid_article",
    "load_wikipedia_dataset",
    "iter_wikipedia_dataset",
]


