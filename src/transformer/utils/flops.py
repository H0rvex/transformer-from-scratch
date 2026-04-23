"""Parameter count and FLOP estimates."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def gpt_block_flops_analytical(num_layers: int, d_model: int, seq_len: int, vocab_size: int | None = None) -> int:
    """
    Rough FLOPs per forward pass for a decoder-only stack (attention + FFN).
    Standard estimate: ~12 * L * d^2 * T per layer (same order as Vaswani-style).
    """
    per_layer = 12 * d_model * d_model * seq_len
    emb = 0
    if vocab_size is not None:
        emb = 2 * seq_len * d_model * vocab_size  # tok + pos lookup + LM head matmul
    return num_layers * per_layer + emb


def fvcore_flops(module: nn.Module, inputs: tuple[Any, ...]) -> int | None:
    if FlopCountAnalysis is None:
        return None
    return int(FlopCountAnalysis(module, inputs).total())
