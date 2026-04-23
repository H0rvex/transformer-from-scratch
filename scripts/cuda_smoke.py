#!/usr/bin/env python3
"""Quick CUDA sanity check; helps debug 'no kernel image for execution on device'."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    if not torch.cuda.is_available():
        print("CUDA not available to PyTorch (driver or CPU-only build).")
        return 1
    d = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device 0: {name}  compute_capability={cap[0]}.{cap[1]}")
    try:
        x = torch.randn(256, 256, device=d, dtype=torch.float32)
        y = x @ x
        torch.cuda.synchronize()
        _ = float(y[0, 0].item())
        print("OK: basic matmul on CUDA succeeded.")
    except RuntimeError as e:
        print("FAILED:", e)
        print()
        print(
            "This usually means your PyTorch wheel was not built for your GPU architecture.\n"
            "Fix: reinstall PyTorch from https://pytorch.org/ — pick the CUDA version that matches\n"
            "your NVIDIA driver (see `nvidia-smi` top-right), not an older/smaller CUDA build by habit.\n"
            "Example (Linux, CUDA 12.1):\n"
            "  pip install --force-reinstall torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "If you are on a very new GPU, you may need a *newer* PyTorch release than your current one."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
