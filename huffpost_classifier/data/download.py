"""Download and prepare the HuffPost dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

from huffpost_classifier.data.preprocess import build_text
from huffpost_classifier.data.splits import create_splits, save_label_map, save_splits


def _write_example_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = {
        "headline": "New study reveals surprising health benefits of walking",
        "short_description": "Researchers found a strong link between daily walks and reduced stress.",
    }
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(example) + "\n")


def download_data(cfg: Any, seed: int) -> Dict[str, Path]:
    """Download HuffPost data, build processed files, and create splits."""
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    splits_path = Path(cfg.data.splits_path)
    examples_path = Path(cfg.data.examples_path)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg.data.dataset_name)
    if isinstance(dataset, dict) and "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset

    def has_headline(example: Dict[str, Any]) -> bool:
        headline = example.get("headline")
        if headline is None:
            return False
        return bool(str(headline).strip())

    full_dataset = full_dataset.filter(has_headline)

    raw_path = raw_dir / "huffpost_raw.jsonl"
    full_dataset.to_json(str(raw_path))

    def build_record(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": build_text(
                example.get("headline", ""),
                example.get("short_description", ""),
                cfg.data.text_sep,
            ),
            "label": example.get("category", "unknown"),
        }

    processed = full_dataset.map(build_record, remove_columns=full_dataset.column_names)
    processed_path = processed_dir / "huffpost_processed.jsonl"
    processed.to_json(str(processed_path))

    labels = sorted(set(processed["label"]))
    save_label_map(labels, processed_dir / "label_map.json")

    splits = create_splits(processed["label"], seed=seed)
    save_splits(splits, splits_path)

    if not examples_path.exists():
        _write_example_file(examples_path)

    return {
        "raw_path": raw_path,
        "processed_path": processed_path,
        "splits_path": splits_path,
        "examples_path": examples_path,
    }
