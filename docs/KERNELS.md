# RMSNorm kernel micro-benchmarks

Device: `cpu`. Mean forward time (ms) over repeated runs.
`TritonRMSNorm` uses the same eager math as :class:`~transformer.models.norm.RMSNorm` (swap-in point for fused Triton).

| Shape | Impl | ms |
|---|---|--:|
| T=256,D=384 | eager_RMSNorm | 0.1363 |
| T=256,D=384 | TritonRMSNorm_py | 0.1138 |
| T=256,D=768 | eager_RMSNorm | 0.1284 |
| T=256,D=768 | TritonRMSNorm_py | 0.1256 |
| T=4096,D=384 | eager_RMSNorm | 1.7431 |
| T=4096,D=384 | TritonRMSNorm_py | 1.6250 |
| T=4096,D=768 | eager_RMSNorm | 4.7581 |
| T=4096,D=768 | TritonRMSNorm_py | 4.5997 |
| T=32768,D=384 | eager_RMSNorm | 31.6502 |
| T=32768,D=384 | TritonRMSNorm_py | 32.9474 |
| T=32768,D=768 | eager_RMSNorm | 66.4326 |
| T=32768,D=768 | TritonRMSNorm_py | 66.5302 |

_Note: CUDA recommended for meaningful latency; rerun on a GPU._

Regenerate: `python scripts/bench_kernels.py`
