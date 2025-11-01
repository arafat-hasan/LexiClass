"""Hugging Face tokenizer plugin."""

from __future__ import annotations

import logging
import os
import warnings
from typing import List

logger = logging.getLogger(__name__)


class HuggingFaceTokenizer:
    """Hugging Face AutoTokenizer wrapper.

    Provides access to all pre-trained tokenizers from Hugging Face.
    Uses fast Rust-based implementation when available.
    Compatible with transformer models (BERT, RoBERTa, GPT, T5, etc.).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        use_fast: bool = True,
        lowercase: bool | None = None,
        add_special_tokens: bool = False,
        return_tokens_only: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
        suppress_warnings: bool = True,
    ) -> None:
        """Initialize Hugging Face tokenizer.

        Args:
            model_name: Model name or path for tokenizer
                - "bert-base-uncased" (BERT, default)
                - "roberta-base" (RoBERTa)
                - "distilbert-base-uncased" (DistilBERT)
                - "albert-base-v2" (ALBERT)
                - "xlnet-base-cased" (XLNet)
                - "t5-base" (T5)
                - etc. (see Hugging Face model hub)
            use_fast: Use fast Rust-based tokenizer if available
            lowercase: Override model's default lowercase setting (None=use model default)
            add_special_tokens: Add special tokens ([CLS], [SEP], etc.)
            return_tokens_only: Return only tokens (strip special tokens in output)
            max_length: Maximum sequence length (None=use model default, 0=unlimited)
            truncation: Truncate sequences longer than max_length
            suppress_warnings: Suppress sequence length warnings (recommended for feature extraction)
        """
        self.model_name = model_name
        self.use_fast = use_fast
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        self.return_tokens_only = return_tokens_only
        self.max_length = max_length
        self.truncation = truncation
        self.suppress_warnings = suppress_warnings
        self.tokenizer = None

    def _load_tokenizer(self) -> None:
        """Lazy load Hugging Face tokenizer."""
        if self.tokenizer is not None:
            return

        # Suppress tokenizer parallelism warnings (occurs with multiprocessing)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Hugging Face tokenizer requires transformers. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading Hugging Face tokenizer: {self.model_name}")

        try:
            # Suppress sequence length warnings if requested
            if self.suppress_warnings:
                # Filter warnings about token sequence length
                warnings.filterwarnings(
                    'ignore',
                    message='.*Token indices sequence length is longer than.*',
                    category=UserWarning,
                )

            # Determine model_max_length
            if self.max_length == 0:
                # Unlimited: use a very large value
                model_max_length = 1_000_000
            elif self.max_length is not None:
                # User-specified max length
                model_max_length = self.max_length
            else:
                # Use model's default (typically 512 for BERT)
                model_max_length = None

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast,
                model_max_length=model_max_length,
            )

            # Override lowercase if specified
            if self.lowercase is not None:
                if hasattr(self.tokenizer, 'do_lower_case'):
                    self.tokenizer.do_lower_case = self.lowercase

            actual_max_len = self.tokenizer.model_max_length
            logger.info(
                f"Tokenizer loaded successfully (vocab_size={self.tokenizer.vocab_size}, "
                f"max_length={actual_max_len})"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer '{self.model_name}': {e}\n"
                f"Make sure the model name is valid or the model is downloaded."
            )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using Hugging Face tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        self._load_tokenizer()

        # Tokenize with or without special tokens
        tokens = self.tokenizer.tokenize(
            text,
            add_special_tokens=self.add_special_tokens
        )

        # Apply truncation if enabled
        if self.truncation and self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Apply lowercase if specified and tokenizer doesn't do it automatically
        if self.lowercase and not getattr(self.tokenizer, 'do_lower_case', False):
            tokens = [token.lower() for token in tokens]

        # Filter special tokens if requested
        if self.return_tokens_only and self.add_special_tokens:
            special_tokens = set(self.tokenizer.all_special_tokens)
            tokens = [token for token in tokens if token not in special_tokens]

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs (useful for transformer models).

        Args:
            text: Input text to encode
            add_special_tokens: Add special tokens

        Returns:
            List of token IDs
        """
        self._load_tokenizer()
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        self._load_tokenizer()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __getstate__(self):
        """Support pickling by storing model name."""
        return {
            'model_name': self.model_name,
            'use_fast': self.use_fast,
            'lowercase': self.lowercase,
            'add_special_tokens': self.add_special_tokens,
            'return_tokens_only': self.return_tokens_only,
            'max_length': self.max_length,
            'truncation': self.truncation,
            'suppress_warnings': self.suppress_warnings,
        }

    def __setstate__(self, state):
        """Restore from pickle and reload tokenizer."""
        self.model_name = state['model_name']
        self.use_fast = state['use_fast']
        self.lowercase = state['lowercase']
        self.add_special_tokens = state['add_special_tokens']
        self.return_tokens_only = state['return_tokens_only']
        self.max_length = state.get('max_length')
        self.truncation = state.get('truncation', False)
        self.suppress_warnings = state.get('suppress_warnings', True)
        self.tokenizer = None
        # Tokenizer will be lazily loaded on first use


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="huggingface",
    display_name="Hugging Face",
    description="Fast Rust-based tokenizers for all transformer models",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["transformers>=4.30"],
    optional_dependencies=["tokenizers>=0.13"],  # For fast tokenizers
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="low",
    requires_pretrained=True,
    pretrained_models=[
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased",
        "albert-base-v2",
        "xlnet-base-cased",
        "t5-base",
    ],
    default_params={
        "model_name": "bert-base-uncased",
        "use_fast": True,
        "add_special_tokens": False,
        "return_tokens_only": True,
        "max_length": None,
        "truncation": False,
        "suppress_warnings": True,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: HuggingFaceTokenizer(**kwargs)
)
