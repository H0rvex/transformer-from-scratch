"""Multi-head attention with SDPA, manual fallback, optional RoPE, GQA, KV cache, ALiBi."""

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

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        inv = self.get_buffer("inv_freq").to(dtype=torch.float32, device=device)
        t = torch.arange(position_offset, position_offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv).to(dtype)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: (B, H, T, D); cos, sin: (1, 1, T, D)."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_alibi_slopes(num_heads: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Slopes per head (H,); geometric decay per ALiBi-style summaries."""
    return torch.tensor(
        [float(2 ** (-8.0 / num_heads * (i + 1))) for i in range(num_heads)],
        device=device,
        dtype=dtype,
    )


class MultiHeadAttention(nn.Module):
    """
    Self-attention: padding mask, causal mask, SDPA, optional RoPE, GQA/MQA,
    optional KV cache, optional ALiBi (decoder-style).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: RotaryEmbedding | None = None,
        num_kv_heads: int | None = None,
        alibi: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        nh_kv = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % nh_kv != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_heads = nh_kv
        self.num_queries_per_kv = num_heads // nh_kv
        self.dropout_p = dropout
        self.rope = rope
        self.use_alibi = alibi

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, nh_kv * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, nh_kv * self.d_k, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        if alibi:
            self.register_buffer("alibi_slopes", build_alibi_slopes(num_heads, torch.device("cpu"), torch.float32))

    def _shape_q(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def _shape_kv(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_kv_heads, self.d_k).transpose(1, 2)

    def _repeat_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.num_kv_heads == self.num_heads:
            return k, v
        r = self.num_queries_per_kv
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
        return k, v

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
        past_kv: tuple[Tensor, Tensor] | None = None,
        rope_position_offset: int = 0,
        cache_len: int = 0,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, _ = x.shape
        q = self._shape_q(self.W_q(x))
        k_new = self._shape_kv(self.W_k(x))
        v_new = self._shape_kv(self.W_v(x))

        if self.rope is not None:
            cos, sin = self.rope(t, x.device, x.dtype, position_offset=rope_position_offset)
            q, k_new = apply_rotary_emb(q, k_new, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            k_cat = torch.cat([pk, k_new], dim=2)
            v_cat = torch.cat([pv, v_new], dim=2)
        else:
            k_cat, v_cat = k_new, v_new

        store_kv = (k_cat, v_cat) if use_cache else None

        k_rep, v_rep = self._repeat_kv(k_cat, v_cat)

        if return_attn_weights:
            assert store_kv is None or not use_cache  # viz path: no cache return mixing
            out_m, w = self._forward_manual(
                q,
                k_rep,
                v_rep,
                attn_mask,
                is_causal,
                cache_len,
                past_kv is not None,
                return_weights=True,
            )
            assert w is not None
            return out_m, w

        sdpa_mask = self._build_sdpa_mask(attn_mask, b, q.shape[2], k_rep.shape[2], q.device, q.dtype)
        causal_use = is_causal and past_kv is None and sdpa_mask is None and not self.use_alibi

        alibi_bias: Tensor | None = None
        if self.use_alibi:
            t_q, t_k = q.shape[2], k_rep.shape[2]
            slopes_buf = self.get_buffer("alibi_slopes")
            slopes = slopes_buf.to(device=q.device, dtype=q.dtype).view(self.num_heads, 1, 1)
            qi = torch.arange(t_q, device=q.device, dtype=torch.long).unsqueeze(1) + cache_len
            kj = torch.arange(t_k, device=q.device, dtype=torch.long).unsqueeze(0)
            dist = kj.to(q.dtype) - qi.to(q.dtype)
            alibi_bias = slopes * dist.unsqueeze(0)

        dropout_p = self.dropout_p if self.training else 0.0
        try:
            if alibi_bias is not None:
                raise RuntimeError("force manual for alibi")
            if past_kv is not None and is_causal:
                raise RuntimeError("cached causal uses manual attention")
            attn_out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=causal_use,
            )
        except Exception:
            out_manual, _ = self._forward_manual(
                q,
                k_rep,
                v_rep,
                attn_mask,
                is_causal,
                cache_len,
                past_kv is not None,
                return_weights=False,
                alibi_bias=alibi_bias,
            )
            if use_cache and store_kv is not None:
                return out_manual, store_kv
            return out_manual

        out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out_o = cast(Tensor, self.W_o(out))
        if use_cache and store_kv is not None:
            return out_o, store_kv
        return out_o

    def _build_sdpa_mask(
        self,
        attn_mask: Tensor | None,
        batch: int,
        t_q: int,
        t_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            if attn_mask.dim() == 4:
                valid = attn_mask
                inv = ~valid
                m = torch.zeros(batch, 1, 1, t_k, device=device, dtype=dtype)
                m.masked_fill_(inv, float("-inf"))
                return m
            return attn_mask.to(dtype=dtype)
        return attn_mask.to(dtype=dtype)

    def _causal_extended_bias(
        self,
        cache_len: int,
        t_q: int,
        t_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        qi = torch.arange(t_q, device=device).unsqueeze(1) + cache_len
        kj = torch.arange(t_k, device=device).unsqueeze(0)
        allowed = kj <= qi
        bias = torch.zeros(t_q, t_k, device=device, dtype=dtype)
        bias.masked_fill_(~allowed, float("-inf"))
        return bias.view(1, 1, t_q, t_k)

    def _forward_manual(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
        cache_len: int,
        has_past: bool,
        return_weights: bool = True,
        alibi_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        b, h, t_q, d = q.shape
        t_k = k.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        if alibi_bias is not None:
            scores = scores + alibi_bias.unsqueeze(0)

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
            if has_past:
                cb = self._causal_extended_bias(cache_len, t_q, t_k, q.device, scores.dtype)
                scores = scores + cb
            else:
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
