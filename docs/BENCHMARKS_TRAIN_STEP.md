# Training-step micro-benchmark (GPT forward + backward + AdamW)

Device: `cpu`. batch=2, seq=64, d_model=128, layers=2, vocab=512.

| Mode | ms/step | tokens/sec (approx) | peak VRAM MB |
|---|---:|---:|---:|
| fp32_no_compile | 11.631 | 11,005 | 0.0 |

Regenerate: `python scripts/benchmark.py --train-step`
