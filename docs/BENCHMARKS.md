# Attention micro-benchmarks

Device: `cuda`. batch=8, seq=128, d_model=256, heads=8.
Forward-only mean latency (ms/step). Tokens/sec derived from SDPA path.

| Implementation | ms/step | tokens/sec (approx) |
|---|---:|---:|
| Ours (SDPA) | 0.543 | 1,886,996 |
| Ours (manual, `return_attn_weights=True`) | 0.766 | — |
| `torch.nn.MultiheadAttention` | 0.608 | — |

Regenerate: `python scripts/benchmark.py`
