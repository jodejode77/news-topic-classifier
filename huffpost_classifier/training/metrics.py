"""Metrics utilities for evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    """Compute confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_confusion_matrix(matrix: np.ndarray, labels: List[str], path: Path) -> None:
    """Plot and save a confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 6))
    im = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(im, ax=axis)
    axis.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="true",
        xlabel="predicted",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > thresh else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)
