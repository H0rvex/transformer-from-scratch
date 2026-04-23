"""CSV metrics logging and optional W&B."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Literal, cast

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class CSVLogger:
    """Append one row per log call."""

    def __init__(self, path: Path | str, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._wrote_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(row)


def maybe_init_wandb(
    project: str,
    name: str | None,
    config: dict[str, Any],
    enabled: bool,
) -> Any:
    if not enabled or wandb is None:
        return None
    raw = os.environ.get("WANDB_MODE", "online")
    valid: set[str] = {"online", "offline", "disabled", "shared"}
    mode = cast(Literal["online", "offline", "disabled", "shared"], raw if raw in valid else "online")
    return wandb.init(project=project, name=name, config=config, mode=mode)


def wandb_log(data: dict[str, Any], step: int | None = None) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def wandb_finish() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()
