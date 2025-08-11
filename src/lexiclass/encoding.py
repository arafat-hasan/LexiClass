from __future__ import annotations

from typing import List

import numpy as np
from sklearn import preprocessing


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


