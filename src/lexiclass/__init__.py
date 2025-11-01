"""LexiClass: Extensible document classification toolkit.

Plugin-based architecture for document classification with support for:
- Multiple tokenizers (ICU, spaCy, SentencePiece, etc.)
- Multiple feature extractors (BoW, TF-IDF, FastText, Sentence-BERT, etc.)
- Multiple classifiers (SVM, XGBoost, Transformers, etc.)

Quick Start with Plugins:
    >>> from lexiclass.plugins import registry
    >>>
    >>> # List available plugins
    >>> tokenizers = registry.list_plugins(plugin_type="tokenizer")
    >>> features = registry.list_plugins(plugin_type="feature_extractor")
    >>> classifiers = registry.list_plugins(plugin_type="classifier")
    >>>
    >>> # Create plugin instances
    >>> tokenizer = registry.create("icu", locale="en")
    >>> feature_extractor = registry.create("bow")
    >>> classifier = registry.create("svm")
    >>>
    >>> # Use with document index and classifier
    >>> from lexiclass.classifier import SVMDocumentClassifier
    >>> from lexiclass.index import DocumentIndex
    >>> from lexiclass.io import DocumentLoader, load_labels
    >>>
    >>> clf = SVMDocumentClassifier(
    ...     tokenizer=tokenizer,
    ...     feature_extractor=feature_extractor
    ... )

Direct imports (for specific components):
    >>> from lexiclass.classifier import SVMDocumentClassifier
    >>> from lexiclass.index import DocumentIndex
    >>> from lexiclass.io import DocumentLoader, load_labels
    >>> from lexiclass.encoding import BinaryClassEncoder, MultiClassEncoder
"""

__version__ = "0.3.0"

from .config import get_settings, Settings
from .logging_utils import configure_logging

# Import plugins to trigger auto-registration
from . import plugins

__all__ = [
    "__version__",
    "get_settings",
    "Settings",
    "configure_logging",
    "plugins",
]


