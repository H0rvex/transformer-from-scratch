from __future__ import annotations

import math

import torch

from transformer.models.positional import SinusoidalPositionalEncoding


def test_sinusoidal_matches_formula() -> None:
    d_model = 64
    max_len = 128
    pe_mod = SinusoidalPositionalEncoding(d_model, max_len=max_len)
    buf = pe_mod.pe[0]  # (max_len, d)
    pos = 7
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    expected_even = torch.sin(pos * div)
    expected_odd = torch.cos(pos * div)
    torch.testing.assert_close(buf[pos, 0::2], expected_even, rtol=0, atol=1e-5)
    torch.testing.assert_close(buf[pos, 1::2], expected_odd, rtol=0, atol=1e-5)
