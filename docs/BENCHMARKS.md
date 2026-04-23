# Attention micro-benchmarks

This file is overwritten by `python scripts/benchmark.py` (CPU or CUDA).

| Implementation | ms/step | tokens/sec (approx) |
|---|---:|---:|
| _Run script to populate_ | — | — |

Regenerate: `python scripts/benchmark.py`

## TensorRT (optional)

After exporting ONNX (`python scripts/export_onnx.py`), you can compile with [TensorRT](https://developer.nvidia.com/tensorrt) for NVIDIA deployment; exact flags depend on your TensorRT version and target GPU.
