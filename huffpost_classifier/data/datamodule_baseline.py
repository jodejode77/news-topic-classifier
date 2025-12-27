"""DataModule for baseline EmbeddingBag model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from huffpost_classifier.data.splits import (
    compute_class_weights,
    compute_sample_weights,
    load_label_map,
    load_splits,
)
from huffpost_classifier.data.vocab import Vocab


class BaselineTextDataset(Dataset):
    """Dataset that returns token ids and label indices."""

    def __init__(self, dataset: Any, vocab: Vocab, label_to_id: Dict[str, int]) -> None:
        self.dataset = dataset
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        row = self.dataset[index]
        token_ids = self.vocab.encode(row["text"])
        label_id = self.label_to_id[row["label"]]
        return token_ids, label_id


class BaselineDataModule(LightningDataModule):
    """LightningDataModule for baseline text classification."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_path = Path(cfg.data.processed_dir) / "huffpost_processed.jsonl"
        self.splits_path = Path(cfg.data.splits_path)
        self.label_map_path = Path(cfg.data.processed_dir) / "label_map.json"
        self.batch_size = int(cfg.model.batch_size)
        self.num_workers = int(cfg.data.num_workers)
        self.use_weighted_sampler = bool(cfg.model.use_weighted_sampler)
        self.vocab: Optional[Vocab] = None
        self.label_map: Optional[Dict[str, Dict[str, int]]] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.train_sampler: Optional[WeightedRandomSampler] = None

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

        vocab = Vocab.build_from_texts(
            train_dataset["text"],
            min_freq=int(self.cfg.model.min_freq),
            max_size=int(self.cfg.model.max_vocab_size),
        )

        self.vocab = vocab
        self.label_map = label_map
        self.class_weights = class_weights
        self.train_dataset = BaselineTextDataset(train_dataset, vocab, label_map["label_to_id"])
        self.val_dataset = BaselineTextDataset(val_dataset, vocab, label_map["label_to_id"])
        self.test_dataset = BaselineTextDataset(test_dataset, vocab, label_map["label_to_id"])

        if self.use_weighted_sampler:
            sample_weights = compute_sample_weights(label_ids, class_weights)
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_batch,
        )

    @staticmethod
    def _apply_limit(indices: List[int], limit: Optional[int]) -> List[int]:
        if limit is None:
            return indices
        return indices[: int(limit)]

    @staticmethod
    def _collate_batch(
        batch: List[Tuple[List[int], int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids_list = []
        offsets = []
        labels = []
        offset = 0
        for token_ids, label_id in batch:
            input_tensor = torch.tensor(token_ids, dtype=torch.long)
            input_ids_list.append(input_tensor)
            offsets.append(offset)
            offset += input_tensor.numel()
            labels.append(label_id)
        input_ids = torch.cat(input_ids_list)
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_ids, offsets_tensor, labels_tensor
