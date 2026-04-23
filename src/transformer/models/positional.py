"""Positional encodings: sinusoidal, learned, optional RoPE lives in attention."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        pe = self.get_buffer("pe")
        return x + pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positions."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) token embeddings
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device)
        return cast(Tensor, x + self.emb(pos).unsqueeze(0).expand(b, -1, -1))
