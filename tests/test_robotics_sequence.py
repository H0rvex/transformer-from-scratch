from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch

from transformer.data.robotics import TrajectorySpec, TrajectoryTokenDataset, generate_trajectory_tokens


def test_trajectory_dataset_shapes() -> None:
    spec = TrajectorySpec(grid_size=8, block_size=12, num_sequences=5, seed=7)
    tokens = generate_trajectory_tokens(spec)
    assert tokens.shape == (5, 13)
    assert int(tokens.max()) < spec.vocab_size
    ds = TrajectoryTokenDataset(tokens)
    x, y = ds[0]
    assert x.shape == (12,)
    assert y.shape == (12,)
    assert torch.equal(x[1:], y[:-1])


def test_robotics_sequence_dry_run_script(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "train_robotics_sequence.py"
    python = shutil.which("python3") or shutil.which("python") or sys.executable
    out_dir = tmp_path / "robotics"
    subprocess.check_call(
        [
            python,
            str(script),
            "--dry-run",
            "--num-sequences",
            "16",
            "--block-size",
            "8",
            "--batch-size",
            "4",
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(repo),
    )
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "best_model.pt").exists()
