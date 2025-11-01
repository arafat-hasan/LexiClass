"""Classifier plugins."""

from __future__ import annotations

# Import all classifier plugins to trigger registration
from . import svm
from . import xgboost
from . import transformer

__all__ = ["svm", "xgboost", "transformer"]
