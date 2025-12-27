"""Baseline EmbeddingBag model."""
from __future__ import annotations

import torch
from torch import nn


class BaselineEmbeddingBagModel(nn.Module):
    """EmbeddingBag + MLP classifier."""

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids, offsets)
        return self.mlp(embedded)
