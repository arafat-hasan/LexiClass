"""Plugin system for LexiClass.

This module provides a plugin registry for tokenizers, feature extractors,
and classifiers. Plugins are automatically registered on import.

Example usage:
    >>> from lexiclass.plugins import registry
    >>> tokenizer = registry.create("icu", locale="en")
    >>> features = registry.create("bow")
    >>> classifier = registry.create("svm")
"""

from __future__ import annotations

# Core plugin infrastructure
from .base import PluginMetadata, PluginRegistration, PluginType
from .registry import PluginRegistry, registry

# Auto-load all plugins by importing their modules
# This triggers the plugin registration decorators/calls at the bottom of each plugin file
from . import tokenizers
from . import features
from . import classifiers

__all__ = [
    "PluginMetadata",
    "PluginRegistration",
    "PluginType",
    "PluginRegistry",
    "registry",
    "tokenizers",
    "features",
    "classifiers",
]
