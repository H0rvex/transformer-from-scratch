"""Multi-head attention with SDPA, manual fallback, optional RoPE."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE frequencies for sequence positions (dim must be even)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        inv = self.get_buffer("inv_freq").to(dtype=torch.float32, device=device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv).to(dtype)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: (B, H, T, D); cos, sin: (1, 1, T, D)."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Self-attention. Supports padding mask, causal mask, SDPA, optional RoPE.
    attn_mask: bool True = valid key position when shape is (B,1,1,T_k) broadcast;
               or float additive (broadcastable to scores).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout
        self.rope = rope

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def _shape(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        q = self._shape(self.W_q(x))
        k = self._shape(self.W_k(x))
        v = self._shape(self.W_v(x))

        if self.rope is not None:
            cos, sin = self.rope(t, x.device, x.dtype)
            q, k = apply_rotary_emb(q, k, cos, sin)

        if return_attn_weights:
            out_m, w = self._forward_manual(q, k, v, attn_mask, is_causal, return_weights=True)
            assert w is not None
            return out_m, w

        sdpa_mask = self._build_sdpa_mask(attn_mask, b, t, q.device, q.dtype)
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        except Exception:
            out_manual, _ = self._forward_manual(q, k, v, attn_mask, is_causal, return_weights=False)
            return out_manual

        out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return cast(Tensor, self.W_o(out))

    def _build_sdpa_mask(
        self,
        attn_mask: Tensor | None,
        batch: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            # True = valid key
            if attn_mask.dim() == 4:
                # (B,1,1,Tk)
                valid = attn_mask
                inv = ~valid
                m = torch.zeros(batch, 1, 1, seq_len, device=device, dtype=dtype)
                m.masked_fill_(inv, float("-inf"))
                return m
            return attn_mask.to(dtype=dtype)
        return attn_mask.to(dtype=dtype)

    def _forward_manual(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
        return_weights: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        b, h, t_q, d = q.shape
        t_k = k.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.shape == (b, 1, 1, t_k):
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                elif attn_mask.shape[-2:] == (t_q, t_k):
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if is_causal:
            causal = torch.triu(
                torch.ones(t_q, t_k, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t_q, self.d_model)
        out = self.W_o(out)
        if return_weights:
            return out, weights
        return out, None
