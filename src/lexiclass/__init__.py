"""LexiClass: Extensible document classification toolkit.

Import concrete functionality from submodules, e.g.:

    from lexiclass.classifier import SVMDocumentClassifier
    from lexiclass.index import DocumentIndex
    from lexiclass.features import FeatureExtractor
    from lexiclass.tokenization import ICUTokenizer
    from lexiclass.io import DocumentLoader, load_labels
    from lexiclass.encoding import BinaryClassEncoder, MultiClassEncoder
"""

__version__ = "0.1.0"

from .config import get_settings, Settings
from .logging_utils import configure_logging

__all__ = [
    "__version__",
    "get_settings",
    "Settings",
    "configure_logging",
]


