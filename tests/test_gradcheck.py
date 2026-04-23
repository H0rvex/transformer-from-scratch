from __future__ import annotations

import torch

from transformer.models.attention import MultiHeadAttention


def test_mha_gradcheck_fp64_cpu() -> None:
    # Small CPU gradcheck is slow but stable on CPU fp64
    d_model, heads = 16, 4
    m = MultiHeadAttention(d_model, heads, dropout=0.0).double().eval()
    x = torch.randn(1, 5, d_model, dtype=torch.double, requires_grad=True)
    mask = torch.ones(1, 1, 1, 5, dtype=torch.bool)

    def fn(z: torch.Tensor) -> torch.Tensor:
        return m(z, attn_mask=mask, is_causal=False, return_attn_weights=False).sum()

    assert torch.autograd.gradcheck(fn, x, eps=1e-6, atol=1e-4, rtol=1e-2)
