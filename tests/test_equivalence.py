from __future__ import annotations

import torch
import torch.nn as nn

from transformer.models.attention import MultiHeadAttention


def test_mha_close_to_torch_multiheadattention() -> None:
    d_model, heads = 64, 4
    batch, seq = 3, 11
    ours = MultiHeadAttention(d_model, heads, dropout=0.0, bias=True)
    ref = nn.MultiheadAttention(d_model, heads, batch_first=True, bias=True)
    d = d_model
    with torch.no_grad():
        ref.in_proj_weight[:d].copy_(ours.W_q.weight)
        ref.in_proj_weight[d : 2 * d].copy_(ours.W_k.weight)
        ref.in_proj_weight[2 * d :].copy_(ours.W_v.weight)
        assert ref.in_proj_bias is not None
        ref.in_proj_bias[:d].copy_(ours.W_q.bias)
        ref.in_proj_bias[d : 2 * d].copy_(ours.W_k.bias)
        ref.in_proj_bias[2 * d :].copy_(ours.W_v.bias)
        ref.out_proj.weight.copy_(ours.W_o.weight)
        ref.out_proj.bias.copy_(ours.W_o.bias)

    x = torch.randn(batch, seq, d_model)
    pad = torch.ones(batch, 1, 1, seq, dtype=torch.bool)
    yo = ours(x, pad, is_causal=False, return_attn_weights=False)
    assert isinstance(yo, torch.Tensor)
    kpm = torch.zeros(batch, seq, dtype=torch.bool)
    yr, _ = ref(x, x, x, need_weights=False, key_padding_mask=kpm)
    assert torch.allclose(yo, yr, atol=5e-5, rtol=1e-4)
