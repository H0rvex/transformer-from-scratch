"""Decoder-only GPT with optional learned PE or RoPE and weight-tied LM head."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from transformer.models.attention import RotaryEmbedding
from transformer.models.init_weights import init_embedding, init_linear, scaled_residual_linear
from transformer.models.layers import DecoderBlock
from transformer.models.positional import LearnedPositionalEmbedding


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        rope = RotaryEmbedding(head_dim) if use_rope else None

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_rope:
            self.pos_emb: nn.Module = nn.Identity()
        else:
            self.pos_emb = LearnedPositionalEmbedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    norm_first=norm_first,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights(num_layers)

    def _init_weights(self, n_layers: int) -> None:
        init_embedding(self.tok_emb)
        for block in self.blocks:
            assert isinstance(block, DecoderBlock)
            init_linear(block.attn.W_q)
            init_linear(block.attn.W_k)
            init_linear(block.attn.W_v)
            scaled_residual_linear(block.attn.W_o, n_layers)
            init_linear(block.ff.lin1)
            scaled_residual_linear(block.ff.lin2, n_layers)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx: Tensor) -> Tensor:
        _b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.block_size}")
        x = self.drop(self.pos_emb(self.tok_emb(idx)))
        for block in self.blocks:
            x = block(x, attn_mask=None)
        x = self.ln_f(x)
        return cast(Tensor, self.lm_head(x))
