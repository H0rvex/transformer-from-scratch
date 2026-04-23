#!/usr/bin/env python3
"""Micro-benchmark: our MHA (SDPA vs manual) vs torch.nn.MultiheadAttention (forward only)."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from transformer.models.attention import MultiHeadAttention


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--out", type=Path, default=Path("docs/BENCHMARKS.md"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, d, h = args.batch, args.seq, args.d_model, args.heads
    x = torch.randn(b, t, d, device=device)
    mask = torch.ones(b, 1, 1, t, dtype=torch.bool, device=device)
    kpm = torch.zeros(b, t, dtype=torch.bool, device=device)

    ours = MultiHeadAttention(d, h, dropout=0.0).to(device).eval()
    torch_mha = nn.MultiheadAttention(d, h, batch_first=True, dropout=0.0).to(device).eval()

    def fwd_ours(manual: bool) -> torch.Tensor:
        if manual:
            out, _ = ours(x, attn_mask=mask, is_causal=False, return_attn_weights=True)
            return out
        out2 = ours(x, attn_mask=mask, is_causal=False, return_attn_weights=False)
        assert isinstance(out2, torch.Tensor)
        return out2

    def fwd_torch() -> torch.Tensor:
        y, _ = torch_mha(x, x, x, need_weights=False, key_padding_mask=kpm)
        return y

    def bench_ms(fn: Callable[[], torch.Tensor]) -> float:
        for _ in range(args.warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            fn()
        _sync()
        return (time.perf_counter() - t0) / args.repeats * 1000

    t_sdpa = bench_ms(lambda: fwd_ours(False))
    t_man = bench_ms(lambda: fwd_ours(True))
    t_torch = bench_ms(fwd_torch)

    tok_per_s = (b * t) / (t_sdpa / 1000)

    lines = [
        "# Attention micro-benchmarks",
        "",
        f"Device: `{device}`. batch={b}, seq={t}, d_model={d}, heads={h}.",
        "Forward-only mean latency (ms/step). Tokens/sec derived from SDPA path.",
        "",
        "| Implementation | ms/step | tokens/sec (approx) |",
        "|---|---:|---:|",
        f"| Ours (SDPA) | {t_sdpa:.3f} | {tok_per_s:,.0f} |",
        f"| Ours (manual, `return_attn_weights=True`) | {t_man:.3f} | — |",
        f"| `torch.nn.MultiheadAttention` | {t_torch:.3f} | — |",
        "",
        "Regenerate: `python scripts/benchmark.py`",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
