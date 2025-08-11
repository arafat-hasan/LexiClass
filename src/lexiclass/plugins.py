from __future__ import annotations

from typing import Callable, Dict, Type

from .features import FeatureExtractor
from .tokenization import ICUTokenizer


class Registry:
    """Simple plugin registry for tokenizers and feature extractors."""

    def __init__(self) -> None:
        self.tokenizers: Dict[str, Callable[..., object]] = {}
        self.features: Dict[str, Callable[..., object]] = {}

    def register_defaults(self) -> "Registry":
        self.tokenizers.setdefault("icu", lambda locale='en': ICUTokenizer(locale))
        self.features.setdefault("bow", lambda: FeatureExtractor())
        return self


registry = Registry().register_defaults()


