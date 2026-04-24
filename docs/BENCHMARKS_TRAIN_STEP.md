# Training-step micro-benchmark (GPT forward + backward + AdamW)

Device: `cuda`. batch=8, seq=128, d_model=256, layers=2, vocab=512.

| Mode | ms/step | tokens/sec (approx) | peak VRAM MB |
|---|---:|---:|---:|
| fp32_compile_off | 8.877 | 115,360 | 78.8 |
| fp16_compile_off | 22.370 | 45,776 | 69.3 |
| bf16_compile_off | 12.704 | 80,605 | 80.3 |

_No `*_compile_on` rows: Inductor uses Triton, which requires CUDA capability ≥ 7.0 (this GPU is `6.1`)._

Regenerate: `python scripts/benchmark.py --train-step`
