"""Vocabulary utilities for the baseline model."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

PUNCT_PATTERN = re.compile(r"[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Simple regex-based tokenizer."""
    lowered = text.lower()
    cleaned = PUNCT_PATTERN.sub(" ", lowered)
    return cleaned.split()


class Vocab:
    """Token-to-id vocabulary with an unknown token."""

    def __init__(self, token_to_id: Dict[str, int], unk_token: str = "<unk>") -> None:
        self.token_to_id = token_to_id
        self.unk_token = unk_token
        self.unk_id = token_to_id[unk_token]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        tokens = tokenize(text)
        if not tokens:
            return [self.unk_id]
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int,
        max_size: int,
        unk_token: str = "<unk>",
    ) -> "Vocab":
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        most_common = [token for token, count in counter.most_common(max_size) if count >= min_freq]
        token_to_id = {unk_token: 0}
        for token in most_common:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, unk_token=unk_token)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump({"token_to_id": self.token_to_id, "unk_token": self.unk_token}, file)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(token_to_id=payload["token_to_id"], unk_token=payload["unk_token"])
