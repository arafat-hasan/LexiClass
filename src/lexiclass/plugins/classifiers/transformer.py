"""Transformer-based classifier plugin."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TransformerClassifier:
    """Transformer-based text classifier using Hugging Face.

    Fine-tunes pre-trained transformers (BERT, RoBERTa, DistilBERT, etc.)
    for document classification. Provides state-of-the-art quality.

    Note: This classifier works directly with text, not pre-computed features.
    It will re-tokenize the input text using the transformer's tokenizer.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        device: str | None = None,
        output_dir: str | None = None,
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 100,
        warmup_steps: int = 500,
    ) -> None:
        """Initialize transformer classifier.

        Args:
            model_name: Pre-trained model name
                - "distilbert-base-uncased" (fast, lightweight - default)
                - "bert-base-uncased" (balanced)
                - "roberta-base" (high quality)
                - "albert-base-v2" (parameter-efficient)
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for fine-tuning
            max_length: Maximum sequence length (tokens)
            device: Device (None=auto, "cuda", "cpu", "mps")
            output_dir: Directory for model checkpoints (temp if None)
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            warmup_steps: Number of warmup steps
        """
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.warmup_steps = warmup_steps

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.is_multilabel = False
        self.is_fitted = False
        self._temp_dir = None

    def train(
        self,
        texts: List[str] | Dict[str, str],
        labels: List[Union[str, List[str]]] | Dict[str, Union[str, List[str]]],
    ) -> "TransformerClassifier":
        """Train transformer classifier.

        Note: Unlike other classifiers, this takes raw text, not feature matrices.

        Args:
            texts: Document texts (list or dict with doc_ids as keys)
            labels: Document labels (list or dict with doc_ids as keys)

        Returns:
            Self for chaining
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Transformer classifier requires transformers and datasets. "
                "Install with: pip install transformers datasets torch"
            )

        from sklearn import preprocessing

        # Convert dict to list if needed
        if isinstance(texts, dict):
            doc_ids = list(texts.keys())
            texts_list = [texts[doc_id] for doc_id in doc_ids]
            labels_list = [labels[doc_id] for doc_id in doc_ids]
        else:
            texts_list = texts
            labels_list = labels

        logger.info(f"Training transformer classifier on {len(texts_list)} documents")

        # Detect multi-label
        self.is_multilabel = isinstance(labels_list[0], list)

        if self.is_multilabel:
            mlb = preprocessing.MultiLabelBinarizer()
            encoded_labels = mlb.fit_transform(labels_list)
            self.label_encoder = mlb
            num_labels = len(mlb.classes_)
            problem_type = "multi_label_classification"
            logger.info(f"Multi-label classification with {num_labels} labels")
        else:
            le = preprocessing.LabelEncoder()
            encoded_labels = le.fit_transform(labels_list)
            self.label_encoder = le
            num_labels = len(le.classes_)
            problem_type = "single_label_classification"
            logger.info(f"Single-label classification with {num_labels} classes")

        # Load tokenizer and model
        logger.info(f"Loading transformer model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type=problem_type,
        )

        # Prepare dataset
        dataset = Dataset.from_dict({
            "text": texts_list,
            "labels": encoded_labels.tolist() if self.is_multilabel else encoded_labels.tolist(),
        })

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Setup output directory
        if self.output_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="lexiclass_transformer_")
            self.output_dir = self._temp_dir
            logger.info(f"Using temporary directory for checkpoints: {self.output_dir}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_strategy="steps",
            report_to="none",  # Disable wandb/tensorboard
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        logger.info("Starting transformer fine-tuning...")
        trainer.train()
        logger.info("Training completed successfully")

        self.is_fitted = True
        return self

    def predict(
        self,
        texts: List[str] | Dict[str, str],
    ) -> Tuple[List[Union[str, List[str]]], np.ndarray]:
        """Predict labels for texts.

        Args:
            texts: Document texts (list or dict)

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        try:
            import torch
        except ImportError:
            raise ImportError("Prediction requires torch. Install with: pip install torch")

        # Convert dict to list if needed
        if isinstance(texts, dict):
            doc_ids = list(texts.keys())
            texts_list = [texts[doc_id] for doc_id in doc_ids]
        else:
            texts_list = texts

        logger.info(f"Predicting labels for {len(texts_list)} documents")

        # Tokenize
        encodings = self.tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device if specified
        if self.device:
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            self.model.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        if self.is_multilabel:
            # Multi-label: sigmoid + threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            binary_preds = (probs > 0.5).astype(int)
            predictions = self.label_encoder.inverse_transform(binary_preds)
            scores = np.max(probs, axis=1)
        else:
            # Single-label: softmax
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
            predictions = self.label_encoder.inverse_transform(pred_indices)
            scores = np.max(probs, axis=1)

        return list(predictions), scores

    def __del__(self):
        """Cleanup temporary directory if created."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Best effort cleanup


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="transformer",
    display_name="Transformer (BERT/RoBERTa)",
    description="Fine-tuned transformer models (state-of-the-art, works with raw text)",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["transformers>=4.30", "torch>=2.0", "datasets>=2.12"],
    supports_streaming=False,
    supports_multilabel=True,
    performance_tier="slow",
    quality_tier="excellent",
    memory_usage="high",
    requires_pretrained=True,
    pretrained_models=[
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "albert-base-v2",
    ],
    default_params={
        "model_name": "distilbert-base-uncased",
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 512,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: TransformerClassifier(**kwargs)
)
