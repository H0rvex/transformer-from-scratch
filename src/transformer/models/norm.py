"""Layer normalization and RMSNorm (eager PyTorch; Triton fused optional)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root mean square normalization (no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_norm(norm_type: str, d_model: int) -> nn.Module:
    """norm_type: layer_norm | rmsnorm | rmsnorm_triton (falls back to rmsnorm if triton unavailable)."""
    nt = norm_type.lower()
    if nt in ("layer_norm", "layernorm", "ln"):
        return nn.LayerNorm(d_model)
    if nt in ("rmsnorm", "rms"):
        return RMSNorm(d_model)
    if nt in ("rmsnorm_triton", "rms_triton"):
        try:
            from transformer.kernels.rmsnorm_triton import TritonRMSNorm  # noqa: PLC0415

            return TritonRMSNorm(d_model)
        except ImportError:
            return RMSNorm(d_model)
    raise ValueError(f"Unknown norm_type: {norm_type}")
