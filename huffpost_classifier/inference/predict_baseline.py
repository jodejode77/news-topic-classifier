"""Inference utilities for the baseline model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch

from huffpost_classifier.data.vocab import Vocab
from huffpost_classifier.models.baseline_model import BaselineEmbeddingBagModel


def _resolve_baseline_dir(artifacts_dir: Path) -> Path:
    candidate = artifacts_dir / "baseline"
    if candidate.exists():
        return candidate
    return artifacts_dir


def load_baseline_artifacts(
    artifacts_dir: Path,
) -> Tuple[BaselineEmbeddingBagModel, Vocab, List[str]]:
    """Load baseline artifacts from disk."""
    base_dir = _resolve_baseline_dir(artifacts_dir)
    model_config_path = base_dir / "model_config.json"
    weights_path = base_dir / "model.pt"
    vocab_path = base_dir / "vocab.json"
    label_map_path = base_dir / "label_map.json"

    with model_config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)
    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)

    vocab = Vocab.load(vocab_path)
    model = BaselineEmbeddingBagModel(
        vocab_size=model_config["vocab_size"],
        embed_dim=model_config["embed_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, vocab, label_map["id_to_label"]


def predict_baseline(
    model: BaselineEmbeddingBagModel, vocab: Vocab, texts: List[str]
) -> torch.Tensor:
    """Return probabilities for input texts."""
    input_ids_list = [torch.tensor(vocab.encode(text), dtype=torch.long) for text in texts]
    offsets = []
    offset = 0
    for input_ids in input_ids_list:
        offsets.append(offset)
        offset += input_ids.numel()
    input_ids = torch.cat(input_ids_list)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, offsets_tensor)
        return torch.softmax(logits, dim=1)
