#!/usr/bin/env python3
r"""
Build a TensorRT engine from ONNX using the ``trtexec`` CLI (no Python TRT bindings required).

Example:

  trtexec --onnx=docs/assets/onnx/classifier.onnx --saveEngine=docs/assets/trt/classifier_fp16.plan \
    --minShapes=input_ids:1x1 --optShapes=input_ids:4x256 --maxShapes=input_ids:8x512 --fp16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Wrapper around trtexec for TensorRT engine build")
    p.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    p.add_argument("--out", type=Path, default=Path("docs/assets/trt/model_fp16.plan"), help="Output .plan path")
    p.add_argument("--input-name", type=str, default="input_ids")
    p.add_argument("--min-b", type=int, default=1)
    p.add_argument("--opt-b", type=int, default=4)
    p.add_argument("--max-b", type=int, default=8)
    p.add_argument("--min-s", type=int, default=1)
    p.add_argument("--opt-s", type=int, default=256)
    p.add_argument("--max-s", type=int, default=512)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--workspace-mb", type=int, default=4096)
    args = p.parse_args()

    trtexec = shutil.which("trtexec")
    if not trtexec:
        print("trtexec not found on PATH (install TensorRT or CUDA TensorRT toolkit).", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.out}",
        f"--minShapes={args.input_name}:{args.min_b}x{args.min_s}",
        f"--optShapes={args.input_name}:{args.opt_b}x{args.opt_s}",
        f"--maxShapes={args.input_name}:{args.max_b}x{args.max_s}",
        f"--memPoolSize=workspace:{args.workspace_mb * (1 << 20)}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
