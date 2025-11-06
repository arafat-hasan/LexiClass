"""LexiClass: Extensible document classification toolkit.

Plugin-based architecture for document classification with support for:
- Multiple tokenizers (ICU, spaCy, SentencePiece, etc.)
- Multiple feature extractors (BoW, TF-IDF, FastText, Sentence-BERT, etc.)
- Multiple classifiers (SVM, XGBoost, Transformers, etc.)

Quick Start (Simple API):
    >>> from lexiclass import DocumentClassifier
    >>>
    >>> # Create classifier from plugins
    >>> clf = DocumentClassifier.from_plugins('icu', 'tfidf', 'svm', tokenizer_locale='en')
    >>>
    >>> # Build index and train
    >>> clf.build_index('./documents', './my_index')
    >>> clf.train('./labels.tsv')
    >>>
    >>> # Save and predict
    >>> clf.save('./model.pkl')
    >>> predictions = clf.predict('./test_documents')

Quick Start (Plugin System):
    >>> from lexiclass.plugins import registry
    >>>
    >>> # List available plugins
    >>> tokenizers = registry.list_plugins(plugin_type="tokenizer")
    >>> features = registry.list_plugins(plugin_type="feature_extractor")
    >>> classifiers = registry.list_plugins(plugin_type="classifier")
    >>>
    >>> # Create plugin instances
    >>> tokenizer = registry.create("icu", locale="en")
    >>> feature_extractor = registry.create("tfidf")
    >>> classifier_plugin = registry.create("svm")

Direct imports (for specific components):
    >>> from lexiclass import DocumentClassifier, DocumentIndex, DocumentLoader, load_labels
"""

__version__ = "0.3.0"

from .config import get_settings, Settings
from .logging_utils import configure_logging

# Import plugins to trigger auto-registration
from . import plugins

# Main high-level API
from .classifier import DocumentClassifier
from .index import DocumentIndex, IndexMetadata
from .io import DocumentLoader, load_labels

__all__ = [
    "__version__",
    "get_settings",
    "Settings",
    "configure_logging",
    "plugins",
    "DocumentClassifier",
    "DocumentIndex",
    "IndexMetadata",
    "DocumentLoader",
    "load_labels",
]


