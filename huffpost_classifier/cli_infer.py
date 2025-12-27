"""Hydra entrypoint for inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import hydra
from datasets import load_dataset
from omegaconf import DictConfig

from huffpost_classifier.data.dvc_utils import ensure_data
from huffpost_classifier.data.preprocess import build_text
from huffpost_classifier.data.splits import load_splits
from huffpost_classifier.inference.predict_baseline import load_baseline_artifacts, predict_baseline
from huffpost_classifier.inference.predict_bert import load_bert_artifacts, predict_bert
from huffpost_classifier.utils.paths import resolve_artifacts_dir_for_model

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


def _load_records_from_jsonl(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                {
                    "headline": payload.get("headline", ""),
                    "short_description": payload.get("short_description", ""),
                }
            )
    return records


def _load_records_from_stdin() -> List[Dict[str, str]]:
    lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
    records: List[Dict[str, str]] = []
    for line in lines:
        payload = json.loads(line)
        records.append(
            {
                "headline": payload.get("headline", ""),
                "short_description": payload.get("short_description", ""),
            }
        )
    return records


def _load_records_from_test(cfg: DictConfig) -> List[Dict[str, str]]:
    raw_path = Path(cfg.data.raw_dir) / "huffpost_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run training once or ensure data is available."
        )
    dataset = load_dataset("json", data_files=str(raw_path), split="train")
    splits = load_splits(Path(cfg.data.splits_path))
    test_indices = splits["test"]
    if cfg.infer.max_rows is not None:
        test_indices = test_indices[: int(cfg.infer.max_rows)]
    test_dataset = dataset.select(test_indices)
    records: List[Dict[str, str]] = []
    for row in test_dataset:
        records.append(
            {
                "headline": row.get("headline", ""),
                "short_description": row.get("short_description", ""),
                "label": row.get("category", ""),
            }
        )
    return records


def _resolve_records(cfg: DictConfig) -> List[Dict[str, str]]:
    if cfg.infer.input_path:
        records = _load_records_from_jsonl(Path(cfg.infer.input_path))
    elif cfg.infer.use_stdin:
        records = _load_records_from_stdin()
    elif cfg.infer.headline:
        records = [
            {"headline": cfg.infer.headline, "short_description": cfg.infer.short_description or ""}
        ]
    else:
        records = _load_records_from_test(cfg)

    return records


def _format_predictions(labels: List[str], probabilities, top_k: int) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for probs in probabilities:
        sorted_indices = probs.argsort(descending=True)
        top_indices = sorted_indices[:top_k].tolist()
        top_k_list = [
            {"label": labels[class_id], "probability": float(probs[class_id])}
            for class_id in top_indices
        ]
        results.append(
            {
                "predicted_label": labels[top_indices[0]],
                "top_k": top_k_list,
            }
        )
    return results


def _print_predictions(predictions: List[Dict[str, object]]) -> None:
    for row_idx, result in enumerate(predictions):
        print(f"prediction[{row_idx}]: {result['predicted_label']}")
        for rank, entry in enumerate(result["top_k"], start=1):
            print(f"  top{rank}: {entry['label']} ({entry['probability']:.4f})")


def _write_predictions(
    path: Path, records: List[Dict[str, str]], predictions: List[Dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record, pred in zip(records, predictions):
            payload = {
                "headline": record.get("headline", ""),
                "short_description": record.get("short_description", ""),
                "text": record.get("text", ""),
                "predicted_label": pred["predicted_label"],
                "top_k": pred["top_k"],
            }
            if record.get("label"):
                payload["true_label"] = record["label"]
            file.write(json.dumps(payload) + "\n")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run inference for the configured model."""
    ensure_data(cfg)

    artifacts_dir = resolve_artifacts_dir_for_model(
        Path(cfg.infer.artifacts_dir),
        Path(cfg.paths.artifacts_dir),
        cfg.model.type,
    )
    if artifacts_dir is None:
        raise FileNotFoundError(
            "No matching artifacts directory found. Train the selected model or set "
            "infer.artifacts_dir=artifacts/<run_name>."
        )

    records = _resolve_records(cfg)
    if cfg.infer.max_rows is not None:
        records = records[: int(cfg.infer.max_rows)]
    texts: List[str] = []
    for record in records:
        text = build_text(
            record.get("headline", ""), record.get("short_description", ""), cfg.data.text_sep
        )
        record["text"] = text
        texts.append(text)

    if cfg.model.type == "baseline_embedding_bag":
        model, vocab, labels = load_baseline_artifacts(artifacts_dir)
        probabilities = predict_baseline(model, vocab, texts)
        predictions = _format_predictions(labels, probabilities, int(cfg.infer.top_k))
        _print_predictions(predictions)
    elif cfg.model.type == "bert_finetune":
        model, tokenizer, labels = load_bert_artifacts(artifacts_dir)
        probabilities = predict_bert(model, tokenizer, texts, int(cfg.model.max_length))
        predictions = _format_predictions(labels, probabilities, int(cfg.infer.top_k))
        _print_predictions(predictions)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    if cfg.infer.output_path:
        _write_predictions(Path(cfg.infer.output_path), records, predictions)


if __name__ == "__main__":
    main()
