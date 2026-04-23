# Attention micro-benchmarks

Device: `cpu`. batch=8, seq=128, d_model=256, heads=8.
Forward-only mean latency (ms/step). Tokens/sec derived from SDPA path.

| Implementation | ms/step | tokens/sec (approx) |
|---|---:|---:|
| Ours (SDPA) | 3.654 | 280,205 |
| Ours (manual, `return_attn_weights=True`) | 5.836 | — |
| `torch.nn.MultiheadAttention` | 7.207 | — |

Regenerate: `python scripts/benchmark.py`
