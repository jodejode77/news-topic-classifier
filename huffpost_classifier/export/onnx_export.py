"""ONNX export utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from huffpost_classifier.inference.predict_baseline import load_baseline_artifacts
from huffpost_classifier.inference.predict_bert import load_bert_artifacts


def export_baseline_onnx(
    artifacts_dir: Path,
    output_path: Path,
    opset: int,
    dynamic_axes: bool,
) -> None:
    """Export baseline model to ONNX."""
    model, _, _ = load_baseline_artifacts(artifacts_dir)
    model.eval()

    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)

    dynamic = None
    if dynamic_axes:
        dynamic = {
            "input_ids": {0: "tokens"},
            "offsets": {0: "batch"},
            "logits": {0: "batch"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["input_ids", "offsets"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic,
        "opset_version": opset,
    }
    try:
        torch.onnx.export(
            model,
            (input_ids, offsets),
            str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            model,
            (input_ids, offsets),
            str(output_path),
            **export_kwargs,
        )


def export_bert_onnx(
    artifacts_dir: Path,
    output_path: Path,
    opset: int,
    dynamic_axes: bool,
    max_length: int,
) -> None:
    """Export TinyBERT model to ONNX."""
    model, _, _ = load_bert_artifacts(artifacts_dir)
    model.eval()

    input_ids = torch.ones((1, max_length), dtype=torch.long)
    attention_mask = torch.ones((1, max_length), dtype=torch.long)

    dynamic = None
    if dynamic_axes:
        dynamic = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic,
        "opset_version": opset,
    }
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(output_path),
            **export_kwargs,
        )


def export_onnx(cfg: Any, artifacts_dir: Path) -> Path:
    """Export ONNX for the configured model type."""
    onnx_dir = artifacts_dir / "onnx"
    if cfg.model.type == "baseline_embedding_bag":
        output_path = onnx_dir / "baseline.onnx"
        export_baseline_onnx(
            artifacts_dir=artifacts_dir,
            output_path=output_path,
            opset=int(cfg.export.opset),
            dynamic_axes=bool(cfg.export.dynamic_axes),
        )
        return output_path
    if cfg.model.type == "bert_finetune":
        output_path = onnx_dir / "bert.onnx"
        export_bert_onnx(
            artifacts_dir=artifacts_dir,
            output_path=output_path,
            opset=int(cfg.export.opset),
            dynamic_axes=bool(cfg.export.dynamic_axes),
            max_length=int(cfg.model.max_length),
        )
        return output_path
    raise ValueError(f"Unknown model type: {cfg.model.type}")
