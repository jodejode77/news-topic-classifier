"""Inference utilities for the TinyBERT model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_bert_dir(artifacts_dir: Path) -> Path:
    candidate = artifacts_dir / "bert"
    if candidate.exists():
        return candidate
    return artifacts_dir


def load_bert_artifacts(
    artifacts_dir: Path,
) -> Tuple[AutoModelForSequenceClassification, Any, List[str]]:
    """Load BERT artifacts from disk."""
    base_dir = _resolve_bert_dir(artifacts_dir)
    model_dir = base_dir / "model"
    tokenizer_dir = base_dir / "tokenizer"
    label_map_path = base_dir / "label_map.json"

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)

    return model, tokenizer, label_map["id_to_label"]


def predict_bert(
    model: AutoModelForSequenceClassification,
    tokenizer: Any,
    texts: List[str],
    max_length: int,
) -> torch.Tensor:
    """Return probabilities for input texts."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        return torch.softmax(outputs.logits, dim=1)
