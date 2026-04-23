# Inference benchmarks

_kind=clf, device=`cpu`, batch=4, seq=128._

| Backend | p50 ms | p95 ms | tokens/s (approx) | peak VRAM MB |
|---|---:|---:|---:|---:|
| pytorch_eager | 1.948 | 2.099 | 262,871 | 0.0 |
| pytorch_compile | 1.843 | 1.905 | 277,747 | 0.0 |
| onnxruntime_cpu | 1.819 | 1.990 | 281,543 | — |
| tensorrt_fp16_plan | — | — | — | — |

**TensorRT:** build a `.plan` with `python scripts/build_trt_engine.py` and benchmark with NVIDIA tooling.

Regenerate: `python scripts/export_onnx.py && python scripts/bench_inference.py --kind clf`
