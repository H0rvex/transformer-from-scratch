from __future__ import annotations

import torch

from transformer.models.attention import MultiHeadAttention


def test_padding_columns_near_zero_mass() -> None:
    d_model, heads = 32, 4
    m = MultiHeadAttention(d_model, heads, dropout=0.0).eval()
    b, t = 1, 6
    x = torch.randn(b, t, d_model)
    mask = torch.ones(b, 1, 1, t, dtype=torch.bool)
    mask[..., 3:] = False  # pad from index 3 onward
    _out, w = m(x, attn_mask=mask, is_causal=False, return_attn_weights=True)
    assert w is not None
    # valid query positions should assign ~0 mass to pad keys
    valid_q = w[0, :, :3, 3:]
    assert valid_q.sum().item() < 5e-3


def test_causal_mask_upper_triangle_zero() -> None:
    d_model, heads = 32, 4
    m = MultiHeadAttention(d_model, heads, dropout=0.0).eval()
    x = torch.randn(1, 5, d_model)
    _out, w = m(x, attn_mask=None, is_causal=True, return_attn_weights=True)
    assert w is not None
    triu = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    wh = w[0, 0]
    assert (wh[triu] < 1e-6).all()
