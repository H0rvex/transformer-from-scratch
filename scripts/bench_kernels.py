#!/usr/bin/env python3
"""Micro-benchmark RMSNorm (eager vs Triton module); writes docs/KERNELS.md."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch

from transformer.kernels.rmsnorm_triton import TritonRMSNorm
from transformer.models.norm import RMSNorm


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_ms(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - t0) / repeats * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/KERNELS.md"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[tuple[str, str, float]] = []

    for ntok in [256, 4096, 32768]:
        for d in [384, 768]:
            x = torch.randn(ntok, d, device=device)
            eager = RMSNorm(d).to(device).eval()
            tr = TritonRMSNorm(d).to(device).eval()

            te = bench_ms(lambda eager=eager, xv=x: eager(xv), args.warmup, args.repeats)
            tt = bench_ms(lambda tr_=tr, xv=x: tr_(xv), args.warmup, args.repeats)
            rows.append((f"T={ntok},D={d}", "eager_RMSNorm", te))
            rows.append((f"T={ntok},D={d}", "TritonRMSNorm_py", tt))

    lines = [
        "# RMSNorm kernel micro-benchmarks",
        "",
        f"Device: `{device}`. Mean forward time (ms) over repeated runs.",
        "`TritonRMSNorm` matches eager :class:`~transformer.models.norm.RMSNorm` (swap-in point for fused Triton).",
        "",
        "| Shape | Impl | ms |",
        "|---|---|--:|",
    ]
    for shape, impl, ms in rows:
        lines.append(f"| {shape} | {impl} | {ms:.4f} |")

    if device.type == "cpu":
        lines.extend(
            [
                "",
                "_Note: CUDA recommended for meaningful latency; rerun on a GPU._",
            ]
        )

    lines.extend(["", "Regenerate: `python scripts/bench_kernels.py`"])
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
