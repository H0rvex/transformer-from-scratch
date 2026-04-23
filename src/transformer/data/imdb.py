"""IMDB sentiment dataloaders and whitespace vocabulary."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ClfBatch = tuple[torch.Tensor, torch.Tensor]


class IMDBDataset(Dataset[ClfBatch]):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._encode(self.texts[idx])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def _encode(self, text: str) -> list[int]:
        words = text.lower().split()
        tokens = [self.vocab.get(w, 1) for w in words]
        tokens = tokens[: self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens


def build_vocab(texts: list[str], vocab_size: int = 20000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text.lower().split())
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(vocab_size))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def get_imdb_dataloaders(
    batch_size: int = 32,
    max_len: int = 256,
    num_workers: int = 0,
    distributed: bool = False,
) -> tuple[DataLoader[ClfBatch], DataLoader[ClfBatch], dict[str, int]]:
    dataset = load_dataset("imdb")
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])
    vocab = build_vocab(train_texts)

    train_ds = IMDBDataset(train_texts, train_labels, vocab, max_len)
    test_ds = IMDBDataset(test_texts, test_labels, vocab, max_len)

    if distributed:
        sampler: DistributedSampler[Any] = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, vocab
