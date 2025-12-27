"""LightningModule wrapper for the baseline model."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from huffpost_classifier.models.baseline_model import BaselineEmbeddingBagModel


class BaselineClassifier(LightningModule):
    """LightningModule for the baseline classifier."""

    def __init__(
        self,
        cfg: Any,
        vocab_size: int,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "class_weights"])
        self.cfg = cfg
        self.model = BaselineEmbeddingBagModel(
            vocab_size=vocab_size,
            embed_dim=int(cfg.model.embed_dim),
            hidden_dim=int(cfg.model.hidden_dim),
            num_classes=num_classes,
            dropout=float(cfg.model.dropout),
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, use_weights: bool
    ) -> torch.Tensor:
        weight = self.class_weights if use_weights else None
        return F.cross_entropy(logits, labels, weight=weight)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input_ids, offsets = batch
        return self.model(input_ids, offsets)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input_ids, offsets, labels = batch
        batch_size = labels.size(0)
        logits = self.model(input_ids, offsets)
        loss = self._compute_loss(logits, labels, use_weights=True)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "train_macro_f1",
            self.train_f1,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        input_ids, offsets, labels = batch
        batch_size = labels.size(0)
        logits = self.model(input_ids, offsets)
        loss = self._compute_loss(logits, labels, use_weights=False)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_macro_f1",
            self.val_f1,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        input_ids, offsets, labels = batch
        batch_size = labels.size(0)
        logits = self.model(input_ids, offsets)
        loss = self._compute_loss(logits, labels, use_weights=False)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, batch_size=batch_size)
        self.log("test_macro_f1", self.test_f1, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.cfg.model.lr))
