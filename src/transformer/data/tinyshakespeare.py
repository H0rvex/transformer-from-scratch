"""TinyShakespeare corpus, BPE training, memmapped token ids, LM dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformer.data.tokenizers import encode_file_to_memmap, load_tokenizer, train_bpe_tokenizer

LmBatch = tuple[torch.Tensor, torch.Tensor]

TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tinyshakespeare(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = data_dir / "input.txt"
    if not raw.exists():
        r = requests.get(TINY_URL, timeout=60)
        r.raise_for_status()
        raw.write_bytes(r.content)
    return raw


class TinyShakespeareDataset(Dataset[LmBatch]):
    """Non-overlapping blocks of length `block_size` for next-token prediction."""

    def __init__(self, memmap_path: Path | str, block_size: int, split: str) -> None:
        self.block_size = block_size
        arr = np.load(str(memmap_path), mmap_mode="r")
        n = int(arr.shape[0])
        split_idx = int(0.9 * n)
        if split == "train":
            self.tokens = arr[:split_idx]
        else:
            self.tokens = arr[split_idx:]
        n = int(self.tokens.shape[0])
        if n <= block_size:
            self._len = 0
        else:
            self._len = (n - block_size - 1) // block_size + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        sl = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(sl[:-1]).long()
        y = torch.from_numpy(sl[1:]).long()
        return x, y


def ensure_tinyshakespeare_artifacts(
    data_dir: Path | str,
    vocab_size: int = 2048,
    force_retrain_tokenizer: bool = False,
) -> tuple[Path, Path, int]:
    """
    Returns (corpus_path, memmap_npy_path, vocab_size_for_model).
    vocab_size_for_model comes from tokenizer.get_vocab_size().
    """
    data_dir = Path(data_dir)
    raw = download_tinyshakespeare(data_dir)
    tok_json = data_dir / "tokenizer.json"
    mem_path = data_dir / "tokens.npy"

    if not tok_json.exists() or force_retrain_tokenizer:
        train_bpe_tokenizer([raw], tok_json, vocab_size=vocab_size)
    tok = load_tokenizer(tok_json)
    vs = tok.get_vocab_size()
    if not mem_path.exists() or force_retrain_tokenizer:
        encode_file_to_memmap(tok, raw, mem_path)
    return raw, mem_path, vs


def get_tinyshakespeare_dataloaders(
    data_dir: Path | str,
    block_size: int,
    batch_size: int,
    vocab_size: int = 2048,
    num_workers: int = 0,
    distributed: bool = False,
) -> tuple[DataLoader[LmBatch], DataLoader[LmBatch], int]:
    _, mem_path, vs = ensure_tinyshakespeare_artifacts(data_dir, vocab_size=vocab_size)
    train_ds = TinyShakespeareDataset(mem_path, block_size, "train")
    val_ds = TinyShakespeareDataset(mem_path, block_size, "val")
    if distributed:
        sampler: DistributedSampler[Any] = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, vs
