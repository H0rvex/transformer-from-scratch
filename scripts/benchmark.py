#!/usr/bin/env python3
"""Micro-benchmark: MHA vs torch.nn.MHA; optional LM training-step sweep (CUDA)."""

from __future__ import annotations

import argparse
import contextlib
import gc
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from transformer.models.attention import MultiHeadAttention
from transformer.models.gpt import GPTModel


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_inductor_ok() -> bool:
    """Default ``torch.compile`` GPU backend (Inductor) uses Triton; Triton needs CUDA capability ≥ 7.0."""
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 7


@torch.no_grad()
def _attention_table(args: argparse.Namespace, out: Path) -> None:
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
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


def _run_train_step_cases(
    device: torch.device,
    batch: int,
    seq: int,
    d_model: int,
    heads: int,
    layers: int,
    vocab: int,
    warmup: int,
    repeats: int,
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []

    dtype_cases: list[tuple[str, torch.dtype]] = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]

    def one_case(label: str, amp_dtype: torch.dtype | None, do_compile: bool) -> tuple[str, float, float, float]:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = GPTModel(
            vocab_size=vocab,
            d_model=d_model,
            num_heads=heads,
            d_ff=d_model * 4,
            num_layers=layers,
            block_size=seq,
            dropout=0.0,
            norm_first=True,
            use_rope=False,
            use_checkpoint=False,
        ).to(device)
        model.train()
        if do_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, vocab, (batch, seq), device=device)
        y = torch.randint(0, vocab, (batch, seq), device=device)
        loss_fn = nn.CrossEntropyLoss()

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if device.type == "cuda" and amp_dtype is not None and amp_dtype != torch.float32
                else contextlib.nullcontext()
            )
            with ctx:
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab), y.view(-1))
            loss.backward()
            opt.step()

        for _ in range(warmup):
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()

        lat: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)

        ms = statistics.mean(lat)
        toks = (batch * seq) / (ms / 1000)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        return label, ms, toks, peak

    if device.type != "cuda":
        label, ms, toks, peak = one_case("fp32_no_compile", None, False)
        rows.append((label, ms, toks, peak))
        return rows

    for dname, dt in dtype_cases:
        if dname == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        rows.append(one_case(f"{dname}_compile_off", dt, False))
        if _cuda_inductor_ok():
            rows.append(one_case(f"{dname}_compile_on", dt, True))

    return rows


def _train_step_table(args: argparse.Namespace, out: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _run_train_step_cases(
        device,
        args.batch,
        args.seq,
        args.d_model,
        args.heads,
        args.layers,
        args.vocab,
        args.warmup,
        args.repeats,
    )
    lines = [
        "# Training-step micro-benchmark (GPT forward + backward + AdamW)",
        "",
        f"Device: `{device}`. batch={args.batch}, seq={args.seq}, d_model={args.d_model}, "
        f"layers={args.layers}, vocab={args.vocab}.",
        "",
        "| Mode | ms/step | tokens/sec (approx) | peak VRAM MB |",
        "|---|---:|---:|---:|",
    ]
    for label, ms, toks, peak in rows:
        lines.append(f"| {label} | {ms:.3f} | {toks:,.0f} | {peak:.1f} |")
    lines.append("")
    if device.type == "cuda" and not _cuda_inductor_ok():
        cap = torch.cuda.get_device_capability()
        lines.append(
            f"_No `*_compile_on` rows: Inductor uses Triton, which requires CUDA capability ≥ 7.0 "
            f"(this GPU is `{cap[0]}.{cap[1]}`)._"
        )
        lines.append("")
    lines.append("Regenerate: `python scripts/benchmark.py --train-step`")
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--train-step", action="store_true", help="LM train-step sweep (CUDA recommended)")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=512)
    args = p.parse_args()

    out = args.out or (Path("docs/BENCHMARKS_TRAIN_STEP.md") if args.train_step else Path("docs/BENCHMARKS.md"))

    if args.train_step:
        _train_step_table(args, out)
    else:
        _attention_table(args, out)


if __name__ == "__main__":
    main()
