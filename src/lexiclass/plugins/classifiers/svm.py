"""SVM classifier plugin."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

from ...config import get_settings

logger = logging.getLogger(__name__)


class SVMClassifier:
    """Linear SVM classifier for document classification.

    Supports binary, multi-class, and multi-label classification.
    Automatically detects the classification type from the labels.
    """

    def __init__(
        self,
        loss: str = 'squared_hinge',
        penalty: str = 'l2',
        tol: float = 1e-5,
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """Initialize SVM classifier.

        Args:
            loss: Loss function ('hinge' or 'squared_hinge')
            penalty: Penalty norm ('l1' or 'l2')
            tol: Tolerance for stopping criterion
            dual: Solve dual or primal optimization problem
            C: Regularization parameter
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.loss = loss
        self.penalty = penalty
        self.tol = tol
        self.dual = dual
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state if random_state is not None else get_settings().random_seed

        self.classifier = None
        self.encoder = None
        self.is_multilabel = False
        self.is_fitted = False

    def train(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
        labels: List[Union[str, List[str]]],
    ) -> "SVMClassifier":
        """Train the SVM classifier.

        Args:
            feature_matrix: Document feature matrix (num_docs x num_features)
            labels: Document labels (string or list of strings for multi-label)

        Returns:
            Self for chaining
        """
        logger.info("Training SVM classifier on %d documents", feature_matrix.shape[0])

        self._setup_encoder_and_classifier(labels)
        encoded_labels = self.encoder.encode(labels)

        logger.info("Fitting SVM model...")
        self.classifier.fit(feature_matrix, encoded_labels)
        self.is_fitted = True

        logger.info("SVM training completed successfully")
        return self

    def predict(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
    ) -> Tuple[List[Union[str, List[str]]], np.ndarray]:
        """Predict labels for feature matrix.

        Args:
            feature_matrix: Document feature matrix

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before prediction")

        logger.info("Predicting labels for %d documents", feature_matrix.shape[0])

        # Get decision function scores
        scores = self.classifier.decision_function(feature_matrix)

        # Decode predictions based on classifier type
        if self.is_multilabel:
            # Multi-label: use encoder's inverse_transform
            binary_predictions = (scores > 0).astype(int)
            predictions = self.encoder.decode(binary_predictions)
            max_scores = np.max(scores, axis=1) if scores.ndim > 1 else scores
        elif hasattr(self.encoder, 'predict'):
            # Binary or multi-class with custom encoder
            predictions = self.encoder.predict(scores)
            if scores.ndim == 1:
                max_scores = scores
            else:
                max_scores = np.max(scores, axis=1)
        else:
            # Should not reach here
            raise ValueError("Encoder type not recognized")

        return list(predictions), max_scores

    def _setup_encoder_and_classifier(self, labels: List[Union[str, List[str]]]) -> None:
        """Setup encoder and classifier based on label structure."""
        sample_label = labels[0]

        if isinstance(sample_label, list):
            # Multi-label classification
            self.is_multilabel = True
            logger.info("Setting up multi-label SVM classifier")

            all_classes: set[str] = set()
            for label_list in labels:
                all_classes.update(label_list)

            self.encoder = preprocessing.MultiLabelBinarizer()
            self.encoder.fit([list(all_classes)])

            self.classifier = OneVsRestClassifier(
                LinearSVC(
                    loss=self.loss,
                    penalty=self.penalty,
                    tol=self.tol,
                    dual=self.dual,
                    C=self.C,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
            )
        else:
            # Single-label classification
            unique_labels = list(set(labels))

            if len(unique_labels) == 2:
                # Binary classification
                logger.info("Setting up binary SVM classifier")
                pos_label, neg_label = unique_labels[0], unique_labels[1]
                self.encoder = BinaryClassEncoder(pos_label, neg_label)
                self.classifier = LinearSVC(
                    loss=self.loss,
                    penalty=self.penalty,
                    tol=self.tol,
                    dual=self.dual,
                    C=self.C,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
            else:
                # Multi-class classification
                logger.info(f"Setting up multi-class SVM classifier ({len(unique_labels)} classes)")
                self.encoder = MultiClassEncoder(unique_labels)
                self.classifier = LinearSVC(
                    loss=self.loss,
                    penalty=self.penalty,
                    tol=self.tol,
                    dual=self.dual,
                    C=self.C,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )



class BinaryClassEncoder:
    """Encoder for binary classification with fixed positive/negative labels."""

    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = -1

    def __init__(self, positive_class: str, negative_class: str) -> None:
        self.positive_class = positive_class
        self.negative_class = negative_class

    def encode(self, labels: List[str]) -> np.ndarray:
        encoded = []
        for label in labels:
            if label == self.positive_class:
                encoded.append(self.POSITIVE_LABEL)
            else:
                encoded.append(self.NEGATIVE_LABEL)
        return np.array(encoded)

    def decode(self, labels: np.ndarray) -> List[str]:
        decoded = []
        for label in labels:
            if label == self.POSITIVE_LABEL:
                decoded.append(self.positive_class)
            else:
                decoded.append(self.negative_class)
        return decoded

    def predict(self, scores: np.ndarray, threshold: float = 0.0) -> List[str]:
        predictions = []
        for score in scores:
            if score > threshold:
                predictions.append(self.positive_class)
            else:
                predictions.append(self.negative_class)
        return predictions


class MultiClassEncoder:
    """Encoder for multi-class classification using scikit-learn LabelEncoder."""

    def __init__(self, classes: List[str]) -> None:
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(classes)

    def encode(self, labels: List[str]) -> np.ndarray:
        return self.encoder.transform(labels)

    def decode(self, labels: np.ndarray) -> List[str]:
        return list(self.encoder.inverse_transform(labels))

    def predict(self, scores: np.ndarray) -> List[str]:
        top_score_indices = np.argmax(scores, axis=1)
        return self.decode(top_score_indices)




# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="svm",
    display_name="Linear SVM",
    description="Linear Support Vector Machine classifier (binary/multi-class/multi-label)",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["scikit-learn>=1.0", "scipy>=1.8", "numpy>=1.22"],
    supports_streaming=False,
    supports_multilabel=True,
    performance_tier="fast",
    quality_tier="good",
    memory_usage="medium",
    default_params={
        "loss": "squared_hinge",
        "penalty": "l2",
        "tol": 1e-5,
        "dual": False,
        "C": 1.0,
        "max_iter": 1000,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: SVMClassifier(**kwargs)
)
