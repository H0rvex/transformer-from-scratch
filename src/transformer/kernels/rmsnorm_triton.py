"""RMSNorm module used when ``norm_type=rmsnorm_triton`` (eager math; Triton fusion in ``bench_kernels``)."""

from __future__ import annotations

from transformer.models.norm import RMSNorm


class TritonRMSNorm(RMSNorm):
    """Same numerics as :class:`~transformer.models.norm.RMSNorm`; swap-in point for fused kernels."""
