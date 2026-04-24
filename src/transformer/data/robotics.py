"""Synthetic robotics-style trajectory token datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

TrajectoryBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class TrajectorySpec:
    grid_size: int = 16
    block_size: int = 32
    num_sequences: int = 1024
    seed: int = 42

    @property
    def vocab_size(self) -> int:
        return self.grid_size * self.grid_size


def _position_to_token(x: int, y: int, grid_size: int) -> int:
    return y * grid_size + x


def generate_trajectory_tokens(spec: TrajectorySpec) -> torch.Tensor:
    """
    Generate simple 2D action-conditioned trajectories as position tokens.

    Each sequence starts at a random grid cell and moves toward a random goal with
    occasional exploration noise. This is not a robotics benchmark; it is a small
    sequence-modeling fixture that resembles discretized state trajectories.
    """
    gen = torch.Generator().manual_seed(spec.seed)
    seqs = torch.empty((spec.num_sequences, spec.block_size + 1), dtype=torch.long)
    for i in range(spec.num_sequences):
        x = int(torch.randint(0, spec.grid_size, (1,), generator=gen).item())
        y = int(torch.randint(0, spec.grid_size, (1,), generator=gen).item())
        gx = int(torch.randint(0, spec.grid_size, (1,), generator=gen).item())
        gy = int(torch.randint(0, spec.grid_size, (1,), generator=gen).item())
        for t in range(spec.block_size + 1):
            seqs[i, t] = _position_to_token(x, y, spec.grid_size)
            if t == spec.block_size:
                break
            noisy = float(torch.rand((), generator=gen).item()) < 0.15
            if noisy:
                dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(torch.randint(0, 4, (1,), generator=gen).item())]
            elif abs(gx - x) >= abs(gy - y) and gx != x:
                dx, dy = (1 if gx > x else -1), 0
            elif gy != y:
                dx, dy = 0, (1 if gy > y else -1)
            else:
                dx, dy = 0, 0
            x = max(0, min(spec.grid_size - 1, x + dx))
            y = max(0, min(spec.grid_size - 1, y + dy))
    return seqs


class TrajectoryTokenDataset(Dataset[TrajectoryBatch]):
    """Next-token prediction over discretized 2D trajectories."""

    def __init__(self, tokens: torch.Tensor) -> None:
        if tokens.dim() != 2 or tokens.size(1) < 2:
            raise ValueError("tokens must have shape (N, T>=2)")
        self.tokens = tokens.long()

    def __len__(self) -> int:
        return int(self.tokens.size(0))

    def __getitem__(self, idx: int) -> TrajectoryBatch:
        seq = self.tokens[idx]
        return seq[:-1], seq[1:]


def get_trajectory_dataloaders(
    spec: TrajectorySpec,
    batch_size: int = 32,
    val_fraction: float = 0.2,
) -> tuple[DataLoader[TrajectoryBatch], DataLoader[TrajectoryBatch]]:
    ds = TrajectoryTokenDataset(generate_trajectory_tokens(spec))
    val_len = max(1, int(len(ds) * val_fraction))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(spec.seed))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
