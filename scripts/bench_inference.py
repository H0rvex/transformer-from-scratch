#!/usr/bin/env python3
r"""
Compare PyTorch eager vs ``torch.compile`` vs ONNX Runtime timings (CUDA when available).

Run ``python scripts/export_onnx.py`` first to produce ONNX under ``docs/assets/onnx/``.
Optional TensorRT: ``python scripts/build_trt_engine.py ...`` then paste numbers from your TRT harness.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


class _OnnxGPTForward(torch.nn.Module):
    """Single-arg forward matching ONNX export."""

    def __init__(self, core: GPTModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.core(input_ids, past_kv_layers=None, use_cache=False, position_offset=0)
        assert isinstance(out, torch.Tensor)
        return out


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_inductor_ok() -> bool:
    """``torch.compile`` on CUDA defaults to Inductor → Triton; Triton needs sm ≥ 7.0."""
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 7


def _percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.array(xs, dtype=np.float64), q))


def bench_pytorch(
    model: torch.nn.Module,
    feed: tuple[torch.Tensor, ...],
    device: torch.device,
    repeats: int,
    warmup: int,
    compile_model: bool,
) -> tuple[list[float], float]:
    m = torch.compile(model, mode="reduce-overhead") if compile_model and hasattr(torch, "compile") else model
    m = m.to(device).eval()

    def one() -> None:
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    m(*feed)
            else:
                m(*feed)

    for _ in range(warmup):
        one()
    _sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lat: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        one()
        _sync()
        lat.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return lat, peak


def bench_onnx(path: Path, feeds: dict[str, np.ndarray], repeats: int, warmup: int, cuda: bool) -> list[float]:
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=prov)
    for _ in range(warmup):
        sess.run(None, feeds)
    lat: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("clf", "gpt"), default="clf")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("docs/INFERENCE_BENCHMARKS.md"))
    p.add_argument("--onnx-dir", type=Path, default=Path("docs/assets/onnx"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"

    if args.kind == "clf":
        model = TransformerClassifier(
            vocab_size=128,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            num_classes=2,
            max_len=args.seq,
        )
        x = torch.randint(0, 128, (args.batch, args.seq), dtype=torch.long)
        feed = (x,)
        onnx_path = args.onnx_dir / "classifier.onnx"
        feeds_np = {"input_ids": x.numpy()}
    else:
        model = _OnnxGPTForward(
            GPTModel(
                vocab_size=256,
                d_model=64,
                num_heads=4,
                d_ff=128,
                num_layers=2,
                block_size=args.seq,
                dropout=0.0,
                norm_first=True,
                use_rope=False,
            )
        )
        xi = torch.randint(0, 256, (args.batch, args.seq), dtype=torch.long)
        feed = (xi,)
        onnx_path = args.onnx_dir / "gpt.onnx"
        feeds_np = {"input_ids": xi.numpy()}

    rows: list[str] = []
    toks_approx = args.batch * args.seq
    skipped_compile = False

    for compile_on in (False, True):
        label = f"pytorch_{'compile' if compile_on else 'eager'}"
        if compile_on and cuda and not _cuda_inductor_ok():
            rows.append("| pytorch_compile | — | — | — | — |")
            skipped_compile = True
            continue
        lat, peak = bench_pytorch(model, feed, device, args.repeats, args.warmup, compile_on)
        p50, p95 = _percentile(lat, 50), _percentile(lat, 95)
        tps = toks_approx / (p50 / 1000) if p50 > 0 else 0.0
        rows.append(f"| {label} | {p50:.3f} | {p95:.3f} | {tps:,.0f} | {peak:.1f} |")

    if onnx_path.exists():
        lat_o = bench_onnx(onnx_path, feeds_np, args.repeats, args.warmup, cuda)
        p50o = _percentile(lat_o, 50)
        tpso = toks_approx / (p50o / 1000) if p50o > 0 else 0.0
        rows.append(
            f"| onnxruntime_{'cuda' if cuda else 'cpu'} | {p50o:.3f} | {_percentile(lat_o, 95):.3f} | {tpso:,.0f} | — |"
        )
    else:
        rows.append("| onnxruntime | — | — | — | — |")

    rows.append("| tensorrt_fp16_plan | — | — | — | — |")

    header = [
        "# Inference benchmarks",
        "",
        f"_kind={args.kind}, device=`{device}`, batch={args.batch}, seq={args.seq}._",
        "",
        "| Backend | p50 ms | p95 ms | tokens/s (approx) | peak VRAM MB |",
        "|---|---:|---:|---:|---:|",
    ]
    footer = [
        "",
        "**TensorRT:** build a `.plan` with `python scripts/build_trt_engine.py` and benchmark with NVIDIA tooling.",
        "",
    ]
    if skipped_compile:
        cap = torch.cuda.get_device_capability()
        footer.append(
            f"**Note:** `pytorch_compile` is omitted (—) on this GPU: Inductor/Triton needs CUDA capability ≥ 7.0; "
            f"device is `{cap[0]}.{cap[1]}`."
        )
        footer.append("")
    footer.append("Regenerate: `python scripts/export_onnx.py && python scripts/bench_inference.py --kind clf`")
    text = "\n".join(header + rows + footer) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
