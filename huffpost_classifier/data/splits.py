"""Dataset split utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def create_splits(labels: List[str], seed: int) -> Dict[str, List[int]]:
    """Create stratified train/val/test splits with fixed seed."""
    indices = np.arange(len(labels))
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )
    return {
        "seed": seed,
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def save_splits(splits: Dict[str, List[int]], path: Path) -> None:
    """Save split indices to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(splits, file)


def load_splits(path: Path) -> Dict[str, List[int]]:
    """Load split indices from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_label_map(labels: List[str], path: Path) -> None:
    """Save label mappings to disk."""
    id_to_label = list(labels)
    label_to_id = {label: index for index, label in enumerate(id_to_label)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"id_to_label": id_to_label, "label_to_id": label_to_id}, file)


def load_label_map(path: Path) -> Dict[str, Dict[str, int]]:
    """Load label mappings from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_class_weights(label_ids: List[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(np.array(label_ids), minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (counts[nonzero] * float(num_classes))
    return torch.tensor(weights, dtype=torch.float)


def compute_sample_weights(label_ids: List[int], class_weights: torch.Tensor) -> List[float]:
    """Map class weights to per-sample weights."""
    return [float(class_weights[label_id]) for label_id in label_ids]
