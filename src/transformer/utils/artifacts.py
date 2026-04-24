"""Run artifact metadata helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip()


def checkpoint_artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "format": "state_dict"}


def write_run_metadata(
    output_dir: Path | str,
    *,
    task: str,
    cfg: DictConfig,
    artifacts: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task": task,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "artifacts": artifacts,
        "summary": summary or {},
        "git_commit": git_commit(),
    }
    path = out / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_run_metadata(path: Path | str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return raw
