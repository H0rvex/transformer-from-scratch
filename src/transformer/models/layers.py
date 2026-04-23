"""Transformer encoder/decoder blocks."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from transformer.models.attention import MultiHeadAttention, RotaryEmbedding


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        if activation == "gelu":
            self.act: nn.Module = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.lin2(self.drop(self.act(self.lin1(x)))))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool = False,
        rope: RotaryEmbedding | None = None,
        ffn_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, rope=rope)
        self.ff = FeedForward(d_model, d_ff, dropout, activation=ffn_activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.norm_first:
            x = x + self.drop1(self.attn(self.norm1(x), attn_mask, is_causal=False))
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop1(self.attn(x, attn_mask, is_causal=False)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class DecoderBlock(nn.Module):
    """Causal self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool = True,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, rope=rope)
        self.ff = FeedForward(d_model, d_ff, dropout, activation="gelu")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.norm_first:
            x = x + self.drop1(self.attn(self.norm1(x), attn_mask, is_causal=True))
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop1(self.attn(x, attn_mask, is_causal=True)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x
