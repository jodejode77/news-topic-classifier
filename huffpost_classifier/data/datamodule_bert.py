"""DataModule for TinyBERT finetuning."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from huffpost_classifier.data.splits import compute_class_weights, load_label_map, load_splits


class BertDataModule(LightningDataModule):
    """LightningDataModule for BERT finetuning."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_path = Path(cfg.data.processed_dir) / "huffpost_processed.jsonl"
        self.splits_path = Path(cfg.data.splits_path)
        self.label_map_path = Path(cfg.data.processed_dir) / "label_map.json"
        self.batch_size = int(cfg.model.batch_size)
        self.num_workers = int(cfg.data.num_workers)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)
        self.label_map: Optional[Dict[str, Dict[str, int]]] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_dataset("json", data_files=str(self.data_path), split="train")
        splits = load_splits(self.splits_path)
        label_map = load_label_map(self.label_map_path)

        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]

        train_indices = self._apply_limit(train_indices, self.cfg.data.limit_train_samples)
        val_indices = self._apply_limit(val_indices, self.cfg.data.limit_val_samples)
        test_indices = self._apply_limit(test_indices, self.cfg.data.limit_test_samples)

        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)

        label_ids = [label_map["label_to_id"][label] for label in train_dataset["label"]]
        class_weights = compute_class_weights(label_ids, num_classes=len(label_map["id_to_label"]))

        def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=int(self.cfg.model.max_length),
            )

        def map_labels(batch: Dict[str, Any]) -> Dict[str, Any]:
            return {"labels": label_map["label_to_id"][batch["label"]]}

        train_dataset = train_dataset.map(tokenize_batch, batched=True)
        val_dataset = val_dataset.map(tokenize_batch, batched=True)
        test_dataset = test_dataset.map(tokenize_batch, batched=True)

        train_dataset = train_dataset.map(map_labels)
        val_dataset = val_dataset.map(map_labels)
        test_dataset = test_dataset.map(map_labels)

        train_dataset = train_dataset.remove_columns(["text", "label"])
        val_dataset = val_dataset.remove_columns(["text", "label"])
        test_dataset = test_dataset.remove_columns(["text", "label"])

        train_dataset.set_format(type="torch")
        val_dataset.set_format(type="torch")
        test_dataset.set_format(type="torch")

        self.label_map = label_map
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.class_weights = class_weights

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _apply_limit(indices, limit: Optional[int]):
        if limit is None:
            return indices
        return indices[: int(limit)]
