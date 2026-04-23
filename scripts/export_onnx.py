#!/usr/bin/env python3
"""Export classifier and GPT to ONNX and verify logits vs PyTorch (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def _onnx_export(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    f: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset_version: int = 17,
) -> None:
    kw: dict[str, Any] = {
        "model": model,
        "args": args,
        "f": str(f),
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
    }
    try:
        torch.onnx.export(**kw, dynamo=False)
    except TypeError:
        torch.onnx.export(**kw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("docs/assets/onnx"))
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clf = TransformerClassifier(
        vocab_size=128,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        num_classes=2,
        max_len=32,
    ).eval()
    x = torch.randint(0, 128, (2, 32), dtype=torch.long)
    clf_path = args.out_dir / "classifier.onnx"
    _onnx_export(
        clf,
        (x,),
        clf_path,
        ["input_ids"],
        ["logits"],
        {"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(str(clf_path))
    sess = ort.InferenceSession(str(clf_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_ids": x.numpy()})[0]
    with torch.no_grad():
        pt_out = clf(x).numpy()
    err = float(np.max(np.abs(ort_out - pt_out)))
    print(f"classifier: max |ONNX - PT| = {err:.6g}")
    assert err < args.atol

    gpt = GPTModel(
        vocab_size=256,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        block_size=32,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    ).eval()
    xi = torch.randint(0, 256, (2, 16), dtype=torch.long)
    gpt_path = args.out_dir / "gpt.onnx"
    _onnx_export(
        gpt,
        (xi,),
        gpt_path,
        ["input_ids"],
        ["logits"],
        {"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
    )
    onnx.checker.check_model(str(gpt_path))
    sess_g = ort.InferenceSession(str(gpt_path), providers=["CPUExecutionProvider"])
    ort_g = sess_g.run(None, {"input_ids": xi.numpy()})[0]
    with torch.no_grad():
        pt_g = gpt(xi).numpy()
    err_g = float(np.max(np.abs(ort_g - pt_g)))
    print(f"gpt: max |ONNX - PT| = {err_g:.6g}")
    assert err_g < args.atol

    print("ONNX export OK.")


if __name__ == "__main__":
    main()
