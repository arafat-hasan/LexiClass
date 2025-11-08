"""XGBoost classifier plugin."""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """XGBoost gradient boosting classifier.

    High-performance gradient boosting that often outperforms
    linear models on document classification tasks.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        objective: str = "auto",
        use_gpu: bool = False,
        random_state: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize XGBoost classifier.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            objective: Loss function (auto-detected from labels if "auto")
            use_gpu: Use GPU acceleration if available
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = None
        self.label_encoder = None
        self.is_multilabel = False
        self.is_fitted = False

    def train(
        self,
        feature_matrix: np.ndarray | sparse.spmatrix,
        labels: List[Union[str, List[str]]],
    ) -> "XGBoostClassifier":
        """Train the XGBoost classifier.

        Args:
            feature_matrix: Document feature matrix (num_docs x num_features)
            labels: Document labels (string or list of strings for multi-label)

        Returns:
            Self for chaining
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost requires xgboost. Install with: pip install xgboost"
            )

        from sklearn import preprocessing

        logger.info("Training XGBoost classifier on %d documents", feature_matrix.shape[0])

        # Detect multi-label
        if isinstance(labels[0], list):
            self.is_multilabel = True
            logger.info("Training XGBoost for multi-label classification")

            # Multi-label: train one classifier per label
            mlb = preprocessing.MultiLabelBinarizer()
            y_encoded = mlb.fit_transform(labels)
            self.label_encoder = mlb

            # Train separate model for each label
            self.model = []
            for i, label in enumerate(mlb.classes_):
                logger.info(f"Training classifier for label: {label}")
                clf = xgb.XGBClassifier(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    objective="binary:logistic",
                    tree_method="gpu_hist" if self.use_gpu else "auto",
                    random_state=self.random_state,
                    **self.kwargs,
                )
                clf.fit(feature_matrix, y_encoded[:, i])
                self.model.append(clf)

        else:
            # Single-label classification
            unique_labels = sorted(set(labels))

            if len(unique_labels) == 2:
                # Binary classification
                logger.info("Training XGBoost for binary classification")
                self.label_encoder = preprocessing.LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(labels)
                objective = "binary:logistic"
            else:
                # Multi-class classification
                logger.info(f"Training XGBoost for {len(unique_labels)}-class classification")
                self.label_encoder = preprocessing.LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(labels)
                objective = "multi:softmax"

            # Override objective if specified
            if self.objective != "auto":
                objective = self.objective

            self.model = xgb.XGBClassifier(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                objective=objective,
                tree_method="gpu_hist" if self.use_gpu else "auto",
                random_state=self.random_state,
                **self.kwargs,
            )
            self.model.fit(feature_matrix, y_encoded)

        self.is_fitted = True
        logger.info("XGBoost training completed successfully")
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

        if self.is_multilabel:
            # Multi-label prediction
            predictions = []
            scores_list = []

            for clf in self.model:
                probs = clf.predict_proba(feature_matrix)[:, 1]
                scores_list.append(probs)

            # Stack scores
            all_scores = np.column_stack(scores_list)

            # Convert to binary predictions (threshold=0.5)
            binary_preds = (all_scores > 0.5).astype(int)

            # Decode to labels
            predictions = self.label_encoder.inverse_transform(binary_preds)

            # Max score per sample
            max_scores = np.max(all_scores, axis=1)

            return list(predictions), max_scores
        else:
            # Single-label prediction
            y_pred = self.model.predict(feature_matrix)

            # Get probabilities for confidence
            probs = self.model.predict_proba(feature_matrix)
            max_probs = np.max(probs, axis=1)

            # Decode labels
            predictions = self.label_encoder.inverse_transform(y_pred.astype(int))

            return list(predictions), max_probs

    def save(self, path: str) -> None:
        """Save XGBoost classifier to disk using pickle.

        Args:
            path: Path to save the classifier
        """
        import pickle

        if not self.is_fitted:
            logger.warning("Saving unfitted XGBoost classifier")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'is_multilabel': self.is_multilabel,
            'is_fitted': self.is_fitted,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'objective': self.objective,
            'use_gpu': self.use_gpu,
            'random_state': self.random_state,
            'kwargs': self.kwargs,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"XGBoost classifier saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load XGBoost classifier from disk.

        Args:
            path: Path to the saved classifier

        Returns:
            Loaded XGBoostClassifier instance
        """
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance with saved parameters
        instance = cls(
            max_depth=model_data['max_depth'],
            learning_rate=model_data['learning_rate'],
            n_estimators=model_data['n_estimators'],
            objective=model_data['objective'],
            use_gpu=model_data['use_gpu'],
            random_state=model_data['random_state'],
            **model_data['kwargs'],
        )

        # Restore state
        instance.model = model_data['model']
        instance.label_encoder = model_data['label_encoder']
        instance.is_multilabel = model_data['is_multilabel']
        instance.is_fitted = model_data['is_fitted']

        logger.info(f"XGBoost classifier loaded from {path}")
        return instance


# Plugin registration
from ..base import PluginMetadata, PluginType
from ..registry import registry

metadata = PluginMetadata(
    name="xgboost",
    display_name="XGBoost",
    description="Gradient boosting classifier (high performance, often better than SVM)",
    plugin_type=PluginType.CLASSIFIER,
    dependencies=["xgboost>=1.7", "scikit-learn>=1.0"],
    optional_dependencies=["cupy>=11.0"],  # For GPU support
    supports_streaming=False,
    supports_multilabel=True,
    performance_tier="fast",
    quality_tier="excellent",
    memory_usage="medium",
    default_params={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "auto",
        "use_gpu": False,
    },
)

registry.register(
    metadata=metadata,
    factory=lambda **kwargs: XGBoostClassifier(**kwargs)
)
