"""SentencePiece tokenizer plugin."""

from __future__ import annotations

import logging
import tempfile
import os
from typing import List, Iterable

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece subword tokenizer.

    Unsupervised tokenizer that learns subword units from text.
    Language-agnostic and handles any Unicode text.
    Used by many transformer models (T5, ALBERT, XLNet, etc.).
    """

    def __init__(
        self,
        model_path: str | None = None,
        vocab_size: int = 8000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        lowercase: bool = False,
    ) -> None:
        """Initialize SentencePiece tokenizer.

        Args:
            model_path: Path to trained model (if None, must call train() first)
            vocab_size: Vocabulary size for training
            model_type: Model type ("unigram", "bpe", "char", or "word")
            character_coverage: Character coverage for training (0.9995 for most, 1.0 for small alphabets)
            lowercase: Convert to lowercase before tokenization
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.lowercase = lowercase
        self.sp = None
        self._trained = False
        self._temp_model = None

    def train(
        self,
        texts: List[str] | Iterable[str],
        output_path: str | None = None
    ) -> "SentencePieceTokenizer":
        """Train SentencePiece model on texts.

        Args:
            texts: Training texts (can be iterator)
            output_path: Where to save model (if None, uses temp file)

        Returns:
            Self for chaining
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "SentencePiece requires sentencepiece. "
                "Install with: pip install sentencepiece"
            )

        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            delete=False,
            suffix='.txt'
        ) as f:
            temp_input = f.name
            logger.info("Writing training data to temporary file...")
            count = 0
            for text in texts:
                if self.lowercase:
                    text = text.lower()
                f.write(text + '\n')
                count += 1
                if count % 10000 == 0:
                    logger.info(f"Wrote {count} texts...")

        logger.info(f"Training SentencePiece model on {count} texts...")

        try:
            # Determine output path
            if output_path is None:
                self._temp_model = tempfile.mktemp(suffix='.model')
                output_path = self._temp_model

            model_prefix = output_path.replace('.model', '')

            # Train model
            spm.SentencePieceTrainer.train(
                input=temp_input,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
            )

            self.model_path = model_prefix + '.model'
            self._load_model()
            self._trained = True
            logger.info(f"SentencePiece model trained successfully (vocab_size={self.vocab_size})")

        finally:
            # Cleanup temporary input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)

        return self

    def _load_model(self) -> None:
        """Load SentencePiece model."""
        if self.sp is not None:
            return

        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")

        if self.model_path is None:
            raise ValueError(
                "No model_path specified. Either provide model_path or call train() first."
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SentencePiece model not found at: {self.model_path}"
            )

        logger.info(f"Loading SentencePiece model from {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        logger.info(f"Model loaded successfully (vocab_size={self.sp.get_piece_size()})")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece.

        Args:
            text: Input text to tokenize

        Returns:
            List of subword tokens
        """
        self._load_model()

        if self.lowercase:
            text = text.lower()

        return self.sp.encode(text, out_type=str)

    def __getstate__(self):
        """Support pickling by storing model path."""
        return {
            'model_path': self.model_path,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'character_coverage': self.character_coverage,
            'lowercase': self.lowercase,
            '_trained': self._trained,
            '_temp_model': self._temp_model,
        }

    def __setstate__(self, state):
        """Restore from pickle and reload model."""
        self.model_path = state['model_path']
        self.vocab_size = state['vocab_size']
        self.model_type = state['model_type']
        self.character_coverage = state['character_coverage']
        self.lowercase = state['lowercase']
        self._trained = state['_trained']
        self._temp_model = state['_temp_model']
        self.sp = None
        # Model will be lazily loaded on first use

    def __del__(self):
        """Cleanup temporary model file if created."""
        if self._temp_model and os.path.exists(self._temp_model):
            try:
                os.unlink(self._temp_model)
                # Also remove .vocab file if it exists
                vocab_file = self._temp_model.replace('.model', '.vocab')
                if os.path.exists(vocab_file):
                    os.unlink(vocab_file)
            except Exception:
                pass  # Best effort cleanup


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="sentencepiece",
    display_name="SentencePiece",
    description="Subword tokenizer for neural models (language-agnostic, trainable)",
    plugin_type=PluginType.TOKENIZER,
    dependencies=["sentencepiece>=0.1.99"],
    supports_streaming=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="low",
    requires_pretrained=False,
    default_params={
        "vocab_size": 8000,
        "model_type": "unigram",
        "character_coverage": 0.9995,
        "lowercase": False,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: SentencePieceTokenizer(**kwargs)
)
