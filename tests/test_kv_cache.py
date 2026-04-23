from __future__ import annotations

import torch

from transformer.models.gpt import GPTModel


def test_cached_matches_full_forward() -> None:
    torch.manual_seed(42)
    vs = 48
    block = 16
    m = GPTModel(
        vocab_size=vs,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        block_size=block,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    ).eval()
    idx = torch.randint(1, vs - 1, (1, 10), dtype=torch.long)

    with torch.no_grad():
        full = m(idx, use_cache=False)
        assert isinstance(full, torch.Tensor)
        logits_full = full[:, -1, :]

        lo, past = m(idx, use_cache=True, position_offset=0)
        assert isinstance(lo, torch.Tensor)
        assert isinstance(past, list)
        logits_cached = lo[:, -1, :]

    torch.testing.assert_close(logits_full, logits_cached, rtol=0, atol=1e-5)
