"""spaCy tokenizer plugin."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class SpacyTokenizer:
    """spaCy-based tokenizer with multilingual support.

    Fast, accurate tokenization using spaCy's linguistic models.
    Supports filtering of punctuation, stop words, and other options.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable: List[str] | None = None,
        lowercase: bool = True,
        remove_punct: bool = True,
        remove_stop: bool = False,
        remove_spaces: bool = True,
    ) -> None:
        """Initialize spaCy tokenizer.

        Args:
            model_name: spaCy model name
                - "en_core_web_sm" (English, small, fast - default)
                - "en_core_web_md" (English, medium, more accurate)
                - "en_core_web_lg" (English, large, most accurate)
                - "xx_ent_wiki_sm" (Multilingual)
                - etc. (see spaCy documentation)
            disable: Pipeline components to disable for speed
                     (default: ["parser", "ner"] for faster tokenization)
            lowercase: Convert tokens to lowercase
            remove_punct: Remove punctuation tokens
            remove_stop: Remove stop words
            remove_spaces: Remove space-only tokens
        """
        self.model_name = model_name
        self.disable = disable if disable is not None else ["parser", "ner"]
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stop = remove_stop
        self.remove_spaces = remove_spaces
        self.nlp = None

    def _load_model(self) -> None:
        """Lazy load spaCy model."""
        if self.nlp is not None:
            return

        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy tokenizer requires spacy. "
                "Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )

        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name, disable=self.disable)
            logger.info(f"spaCy model loaded successfully")
        except OSError:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Download with: python -m spacy download {self.model_name}"
            )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        self._load_model()

        doc = self.nlp(text)  # type: ignore[misc]
        tokens = []

        for token in doc:
            # Apply filters
            if self.remove_spaces and token.is_space:
                continue
            if self.remove_punct and token.is_punct:
                continue
            if self.remove_stop and token.is_stop:
                continue

            # Get token text
            text_token = token.text
            if self.lowercase:
                text_token = text_token.lower()

            tokens.append(text_token)

        return tokens

    def __getstate__(self):
        """Support pickling by not serializing the nlp model."""
        return {
            'model_name': self.model_name,
            'disable': self.disable,
            'lowercase': self.lowercase,
            'remove_punct': self.remove_punct,
            'remove_stop': self.remove_stop,
            'remove_spaces': self.remove_spaces,
        }

    def __setstate__(self, state):
        """Restore from pickle and reload model."""
        self.model_name = state['model_name']
        self.disable = state['disable']
        self.lowercase = state['lowercase']
        self.remove_punct = state['remove_punct']
        self.remove_stop = state['remove_stop']
        self.remove_spaces = state['remove_spaces']
        self.nlp = None
        # Model will be lazily loaded on first use

    def save(self, path: str) -> None:
        """Save tokenizer to disk using pickle.

        Args:
            path: Path to save the tokenizer
        """
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"SpacyTokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SpacyTokenizer":
        """Load tokenizer from disk.

        Args:
            path: Path to the saved tokenizer

        Returns:
            Loaded SpacyTokenizer instance
        """
        import pickle

        with open(path, 'rb') as f:
            instance = pickle.load(f)

        logger.info(f"SpacyTokenizer loaded from {path}")
        return instance


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="spacy",
    display_name="spaCy",
    description="Modern multilingual tokenizer with linguistic features (fast, high quality)",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["spacy>=3.0"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="medium",
    requires_pretrained=True,
    pretrained_models=["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "xx_ent_wiki_sm"],
    default_params={
        "model_name": "en_core_web_sm",
        "lowercase": True,
        "remove_punct": True,
        "remove_stop": False,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: SpacyTokenizer(**kwargs)
)
