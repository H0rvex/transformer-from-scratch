"""Cosine LR schedule with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            progress = step / max(1, total_steps - 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
