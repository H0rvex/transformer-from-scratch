from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def _python() -> str:
    return shutil.which("python3") or shutil.which("python") or sys.executable


def test_evaluate_classifier_synthetic_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = {
        "model": {
            "vocab_size": 4,
            "d_model": 16,
            "num_heads": 4,
            "d_ff": 32,
            "num_layers": 1,
            "num_classes": 2,
            "max_len": 4,
        },
        "data": {"max_len": 4, "batch_size": 2, "num_workers": 0},
        "train": {},
    }
    ckpt = tmp_path / "classifier.pt"
    torch.save(
        TransformerClassifier(4, 16, 4, 32, 1, 2, dropout=0.0, max_len=4).state_dict(),
        ckpt,
    )
    meta = tmp_path / "run_metadata.json"
    meta.write_text(json.dumps({"task": "classifier", "config": cfg, "artifacts": {}, "summary": {}}), encoding="utf-8")
    out = tmp_path / "metrics.json"
    subprocess.check_call(
        [
            _python(),
            str(repo / "scripts" / "evaluate.py"),
            "--task",
            "classifier",
            "--checkpoint",
            str(ckpt),
            "--metadata",
            str(meta),
            "--synthetic-smoke",
            "--out",
            str(out),
        ],
        cwd=str(repo),
    )
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["task"] == "classifier"
    assert "accuracy" in metrics


def test_evaluate_gpt_synthetic_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = {
        "model": {
            "vocab_size": 16,
            "d_model": 16,
            "num_heads": 4,
            "d_ff": 32,
            "num_layers": 1,
            "block_size": 8,
            "norm_first": True,
        },
        "data": {"block_size": 8, "batch_size": 2, "num_workers": 0},
        "train": {},
    }
    ckpt = tmp_path / "gpt.pt"
    torch.save(GPTModel(16, 16, 4, 32, 1, 8, dropout=0.0).state_dict(), ckpt)
    meta = tmp_path / "run_metadata.json"
    meta.write_text(json.dumps({"task": "gpt", "config": cfg, "artifacts": {}, "summary": {}}), encoding="utf-8")
    out = tmp_path / "metrics.json"
    subprocess.check_call(
        [
            _python(),
            str(repo / "scripts" / "evaluate.py"),
            "--task",
            "gpt",
            "--checkpoint",
            str(ckpt),
            "--metadata",
            str(meta),
            "--synthetic-smoke",
            "--out",
            str(out),
        ],
        cwd=str(repo),
    )
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["task"] == "gpt"
    assert metrics["val_loss"] > 0
