from __future__ import annotations

import torch

from transformer.models.attention import MultiHeadAttention


def test_gqa_matches_mha_when_kv_heads_equal_heads() -> None:
    d_model, heads = 64, 4
    b, t = 2, 9
    x = torch.randn(b, t, d_model)
    mask = torch.ones(b, 1, 1, t, dtype=torch.bool)

    m1 = MultiHeadAttention(d_model, heads, dropout=0.0, num_kv_heads=None)
    m2 = MultiHeadAttention(d_model, heads, dropout=0.0, num_kv_heads=heads)
    with torch.no_grad():
        m2.load_state_dict(m1.state_dict())

    y1 = m1(x, attn_mask=mask, is_causal=False, return_attn_weights=False)
    y2 = m2(x, attn_mask=mask, is_causal=False, return_attn_weights=False)
    assert isinstance(y1, torch.Tensor) and isinstance(y2, torch.Tensor)
    torch.testing.assert_close(y1, y2, rtol=0, atol=1e-6)
