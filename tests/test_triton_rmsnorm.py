from __future__ import annotations

import pytest
import torch

from transformer.kernels.rmsnorm_triton import TritonRMSNorm
from transformer.models.norm import RMSNorm


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _cuda_usable(), reason="CUDA-only parity check (needs compatible GPU + PyTorch build)")
def test_triton_rmsnorm_matches_eager_fwd() -> None:
    d = 64
    x = torch.randn(4, 128, d, device="cuda")
    eager = RMSNorm(d).cuda().eval()
    tr = TritonRMSNorm(d).cuda().eval()
    with torch.no_grad():
        tr.weight.copy_(eager.weight)
        ye = eager(x)
        yt = tr(x)
    torch.testing.assert_close(ye, yt, rtol=0, atol=1e-5)
