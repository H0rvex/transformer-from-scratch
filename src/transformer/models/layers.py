"""Transformer encoder/decoder blocks."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.models.attention import MultiHeadAttention, RotaryEmbedding
from transformer.models.norm import build_norm


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.activation = activation.lower()
        self.drop = nn.Dropout(dropout)

        self.lin1: nn.Linear | None = None
        self.lin2: nn.Linear | None = None
        self.act: nn.Module | None = None
        self.w_gate: nn.Linear | None = None
        self.w_up: nn.Linear | None = None
        self.w_down: nn.Linear | None = None

        if self.activation == "swiglu":
            self.w_gate = nn.Linear(d_model, d_ff)
            self.w_up = nn.Linear(d_model, d_ff)
            self.w_down = nn.Linear(d_ff, d_model)
        else:
            self.lin1 = nn.Linear(d_model, d_ff)
            self.lin2 = nn.Linear(d_ff, d_model)
            if self.activation == "gelu":
                self.act = nn.GELU()
            else:
                self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            assert self.w_gate is not None and self.w_up is not None and self.w_down is not None
            inner = F.silu(self.w_gate(x)) * self.w_up(x)
            return cast(Tensor, self.w_down(self.drop(inner)))
        assert self.lin1 is not None and self.lin2 is not None and self.act is not None
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
        norm_type: str = "layer_norm",
        num_kv_heads: int | None = None,
        alibi: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            rope=rope,
            num_kv_heads=num_kv_heads,
            alibi=alibi,
        )
        self.ff = FeedForward(d_model, d_ff, dropout, activation=ffn_activation)
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            def _bk(x_: Tensor, m_: Tensor | None) -> Tensor:
                return self._forward_inner(x_, m_)

            return cast(Tensor, checkpoint(_bk, x, attn_mask, use_reentrant=False))
        return self._forward_inner(x, attn_mask)

    def _forward_inner(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
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
        ffn_activation: str = "gelu",
        norm_type: str = "layer_norm",
        num_kv_heads: int | None = None,
        alibi: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            rope=rope,
            num_kv_heads=num_kv_heads,
            alibi=alibi,
        )
        self.ff = FeedForward(d_model, d_ff, dropout, activation=ffn_activation)
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        past_kv: tuple[Tensor, Tensor] | None = None,
        rope_position_offset: int = 0,
        cache_len: int = 0,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.use_checkpoint and self.training and past_kv is None and not use_cache:
            from torch.utils.checkpoint import checkpoint

            def _bk(
                x_: Tensor,
                m_: Tensor | None,
            ) -> Tensor:
                out_inner = self._forward_inner(x_, m_, None, rope_position_offset, cache_len, False)
                assert isinstance(out_inner, Tensor)
                return out_inner

            return cast(Tensor, checkpoint(_bk, x, attn_mask, use_reentrant=False))

        return self._forward_inner(x, attn_mask, past_kv, rope_position_offset, cache_len, use_cache)

    def _forward_inner(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        past_kv: tuple[Tensor, Tensor] | None,
        rope_position_offset: int,
        cache_len: int,
        use_cache: bool,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        present: tuple[Tensor, Tensor] | None = None

        if self.norm_first:
            raw = self.attn(
                self.norm1(x),
                attn_mask,
                is_causal=True,
                past_kv=past_kv,
                rope_position_offset=rope_position_offset,
                cache_len=cache_len,
                use_cache=use_cache,
            )
            if use_cache and isinstance(raw, tuple):
                attn_out, present = raw
            else:
                assert isinstance(raw, Tensor)
                attn_out = raw
            x = x + self.drop1(attn_out)
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            raw = self.attn(
                x,
                attn_mask,
                is_causal=True,
                past_kv=past_kv,
                rope_position_offset=rope_position_offset,
                cache_len=cache_len,
                use_cache=use_cache,
            )
            if use_cache and isinstance(raw, tuple):
                attn_out, present = raw
            else:
                assert isinstance(raw, Tensor)
                attn_out = raw
            x = self.norm1(x + self.drop1(attn_out))
            x = self.norm2(x + self.drop2(self.ff(x)))

        if use_cache and present is not None:
            return x, present
        return x
