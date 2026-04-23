"""Weight initialization helpers."""

from __future__ import annotations

import math

import torch.nn as nn


def init_linear(m: nn.Linear) -> None:
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_embedding(m: nn.Embedding) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=0.02)


def scaled_residual_linear(m: nn.Linear, n_layers: int, base_std: float = 0.02) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=base_std / math.sqrt(2 * n_layers))
    if m.bias is not None:
        nn.init.zeros_(m.bias)
