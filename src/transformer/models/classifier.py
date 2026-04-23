"""Encoder-only transformer for sequence classification."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from transformer.models.attention import RotaryEmbedding
from transformer.models.init_weights import init_embedding, init_linear, scaled_residual_linear
from transformer.models.layers import EncoderBlock
from transformer.models.positional import LearnedPositionalEmbedding, SinusoidalPositionalEncoding


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        max_len: int = 512,
        norm_first: bool = False,
        pos_encoding: str = "sinusoidal",  # sinusoidal | learned | none
        use_rope: bool = False,
        ffn_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        head_dim = d_model // num_heads
        rope = RotaryEmbedding(head_dim) if use_rope else None
        if use_rope or pos_encoding == "none":
            self.pos_enc: nn.Module = nn.Identity()
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEmbedding(max_len, d_model)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    norm_first=norm_first,
                    rope=rope,
                    ffn_activation=ffn_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights(num_layers)

    def _init_weights(self, n_layers: int) -> None:
        init_embedding(self.embedding)
        for layer in self.layers:
            assert isinstance(layer, EncoderBlock)
            init_linear(layer.attn.W_q)
            init_linear(layer.attn.W_k)
            init_linear(layer.attn.W_v)
            scaled_residual_linear(layer.attn.W_o, n_layers)
            init_linear(layer.ff.lin1)
            scaled_residual_linear(layer.ff.lin2, n_layers)
        init_linear(self.classifier)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        attn_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        pool_mask = (x != 0).unsqueeze(-1).float()

        out = self.dropout(self.pos_enc(self.embedding(x)))
        for layer in self.layers:
            assert isinstance(layer, EncoderBlock)
            out = layer(out, attn_mask)
        out = self.dropout(out)
        out = (out * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)
        return cast(Tensor, self.classifier(out))
