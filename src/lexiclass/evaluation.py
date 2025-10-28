"""Evaluation metrics and utilities for document classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.

    Attributes:
        accuracy: Overall accuracy score
        per_class: Dict mapping class labels to their metrics (precision, recall, f1, support)
        macro_avg: Macro-averaged metrics across all classes
        weighted_avg: Weighted-averaged metrics (weighted by support)
        confusion_matrix: Confusion matrix as 2D list
        class_labels: Ordered list of class labels
        num_samples: Total number of samples evaluated
        num_missing_predictions: Number of ground truth samples without predictions
        num_missing_ground_truth: Number of predictions without ground truth
    """
    accuracy: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro_avg: Dict[str, float] = field(default_factory=dict)
    weighted_avg: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)
    num_samples: int = 0
    num_missing_predictions: int = 0
    num_missing_ground_truth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'accuracy': self.accuracy,
            'per_class': self.per_class,
            'macro_avg': self.macro_avg,
            'weighted_avg': self.weighted_avg,
            'confusion_matrix': self.confusion_matrix,
            'class_labels': self.class_labels,
            'num_samples': self.num_samples,
            'num_missing_predictions': self.num_missing_predictions,
            'num_missing_ground_truth': self.num_missing_ground_truth,
        }


def load_predictions(filepath: str | Path) -> Dict[str, Tuple[str, float]]:
    """Load predictions from TSV file.

    Expected format: doc_id<TAB>label<TAB>score

    Args:
        filepath: Path to predictions TSV file

    Returns:
        Dictionary mapping doc_id to (label, score) tuple

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    predictions: Dict[str, Tuple[str, float]] = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid format at line {line_num}: expected 3 columns (doc_id, label, score), "
                    f"got {len(parts)}"
                )

            doc_id, label, score_str = parts
            try:
                score = float(score_str)
            except ValueError:
                raise ValueError(
                    f"Invalid score at line {line_num}: '{score_str}' is not a valid float"
                )

            predictions[doc_id] = (label, score)

    logger.info("Loaded %d predictions from %s", len(predictions), filepath)
    return predictions


def load_ground_truth(filepath: str | Path) -> Dict[str, str]:
    """Load ground truth labels from TSV file.

    Expected format: doc_id<TAB>label

    Args:
        filepath: Path to ground truth TSV file

    Returns:
        Dictionary mapping doc_id to label

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    ground_truth: Dict[str, str] = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid format at line {line_num}: expected 2 columns (doc_id, label), "
                    f"got {len(parts)}"
                )

            doc_id, label = parts
            ground_truth[doc_id] = label

    logger.info("Loaded %d ground truth labels from %s", len(ground_truth), filepath)
    return ground_truth


def evaluate_predictions(
    predictions: Dict[str, Tuple[str, float]],
    ground_truth: Dict[str, str],
) -> EvaluationMetrics:
    """Calculate evaluation metrics by comparing predictions with ground truth.

    Args:
        predictions: Dictionary mapping doc_id to (predicted_label, score)
        ground_truth: Dictionary mapping doc_id to true_label

    Returns:
        EvaluationMetrics object containing all calculated metrics

    Raises:
        ValueError: If no matching documents found between predictions and ground truth
    """
    # Find intersection of doc_ids
    pred_ids = set(predictions.keys())
    truth_ids = set(ground_truth.keys())
    common_ids = pred_ids & truth_ids

    num_missing_predictions = len(truth_ids - pred_ids)
    num_missing_ground_truth = len(pred_ids - truth_ids)

    if not common_ids:
        raise ValueError(
            "No matching documents found between predictions and ground truth. "
            f"Predictions: {len(pred_ids)} docs, Ground truth: {len(truth_ids)} docs"
        )

    if num_missing_predictions > 0:
        logger.warning(
            "%d documents in ground truth have no predictions (will be excluded)",
            num_missing_predictions
        )

    if num_missing_ground_truth > 0:
        logger.warning(
            "%d predictions have no ground truth label (will be excluded)",
            num_missing_ground_truth
        )

    # Extract labels for matching documents
    y_true = []
    y_pred = []
    for doc_id in sorted(common_ids):
        y_true.append(ground_truth[doc_id])
        y_pred.append(predictions[doc_id][0])  # Get label from (label, score) tuple

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique labels (sorted for consistent ordering)
    labels = sorted(set(y_true) | set(y_pred))

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Build per-class metrics dictionary
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
        }

    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )

    macro_avg = {
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'f1_score': float(macro_f1),
    }

    weighted_avg = {
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1_score': float(weighted_f1),
    }

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_mat = cm.tolist()

    logger.info("Evaluation complete: accuracy=%.4f on %d samples", accuracy, len(common_ids))

    return EvaluationMetrics(
        accuracy=accuracy,
        per_class=per_class,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
        confusion_matrix=confusion_mat,
        class_labels=labels,
        num_samples=len(common_ids),
        num_missing_predictions=num_missing_predictions,
        num_missing_ground_truth=num_missing_ground_truth,
    )


def format_results_text(metrics: EvaluationMetrics, show_confusion_matrix: bool = False) -> str:
    """Format evaluation metrics as human-readable text.

    Args:
        metrics: EvaluationMetrics object
        show_confusion_matrix: Whether to include confusion matrix in output

    Returns:
        Formatted string for console display
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Evaluation Results")
    lines.append("=" * 70)
    lines.append("")

    # Dataset statistics
    lines.append("Dataset Statistics:")
    lines.append(f"  Total samples evaluated: {metrics.num_samples:,}")
    if metrics.num_missing_predictions > 0:
        lines.append(f"  Missing predictions: {metrics.num_missing_predictions:,}")
    if metrics.num_missing_ground_truth > 0:
        lines.append(f"  Missing ground truth: {metrics.num_missing_ground_truth:,}")
    lines.append("")

    # Overall accuracy
    lines.append("Overall Metrics:")
    lines.append(f"  Accuracy: {metrics.accuracy:.4f}")
    lines.append("")

    # Per-class metrics
    lines.append("Per-Class Metrics:")
    header = f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    lines.append(header)
    lines.append("-" * 70)

    for label in metrics.class_labels:
        m = metrics.per_class[label]
        line = (
            f"{label:<20} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['support']:>10,}"
        )
        lines.append(line)

    lines.append("-" * 70)

    # Macro average
    line = (
        f"{'Macro Average':<20} "
        f"{metrics.macro_avg['precision']:>10.4f} "
        f"{metrics.macro_avg['recall']:>10.4f} "
        f"{metrics.macro_avg['f1_score']:>10.4f} "
        f"{metrics.num_samples:>10,}"
    )
    lines.append(line)

    # Weighted average
    line = (
        f"{'Weighted Average':<20} "
        f"{metrics.weighted_avg['precision']:>10.4f} "
        f"{metrics.weighted_avg['recall']:>10.4f} "
        f"{metrics.weighted_avg['f1_score']:>10.4f} "
        f"{metrics.num_samples:>10,}"
    )
    lines.append(line)
    lines.append("")

    # Confusion matrix (optional)
    if show_confusion_matrix:
        lines.append("Confusion Matrix:")
        lines.append("(Rows: True labels, Columns: Predicted labels)")

        # Header with class labels
        header = " " * 20 + "  ".join(f"{label:>8}" for label in metrics.class_labels)
        lines.append(header)

        for i, true_label in enumerate(metrics.class_labels):
            row = f"{true_label:<20}"
            for j in range(len(metrics.class_labels)):
                row += f"  {metrics.confusion_matrix[i][j]:>8,}"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def format_results_json(metrics: EvaluationMetrics) -> str:
    """Format evaluation metrics as JSON.

    Args:
        metrics: EvaluationMetrics object

    Returns:
        JSON string
    """
    return json.dumps(metrics.to_dict(), indent=2)


def format_results_tsv(metrics: EvaluationMetrics) -> str:
    """Format evaluation metrics as TSV (tab-separated values).

    Args:
        metrics: EvaluationMetrics object

    Returns:
        TSV formatted string
    """
    lines = []

    # Header
    lines.append("label\tprecision\trecall\tf1_score\tsupport")

    # Per-class metrics
    for label in metrics.class_labels:
        m = metrics.per_class[label]
        lines.append(
            f"{label}\t{m['precision']:.4f}\t{m['recall']:.4f}\t"
            f"{m['f1_score']:.4f}\t{m['support']}"
        )

    # Averages
    lines.append(
        f"macro_avg\t{metrics.macro_avg['precision']:.4f}\t"
        f"{metrics.macro_avg['recall']:.4f}\t{metrics.macro_avg['f1_score']:.4f}\t"
        f"{metrics.num_samples}"
    )

    lines.append(
        f"weighted_avg\t{metrics.weighted_avg['precision']:.4f}\t"
        f"{metrics.weighted_avg['recall']:.4f}\t{metrics.weighted_avg['f1_score']:.4f}\t"
        f"{metrics.num_samples}"
    )

    return "\n".join(lines)
