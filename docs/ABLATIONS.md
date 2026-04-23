# Ablations (synthetic sanity)

Short runs on a fixed synthetic binary dataset (CPU) to compare **pre-LN vs post-LN**,
**positional encoding variants**, **RoPE**, **RMSNorm**, **SwiGLU**, and **GQA** at tiny width.

| setting | val accuracy |
|---|---:|
| post_ln_sinusoidal | 0.625 |
| pre_ln_sinusoidal | 0.375 |
| pre_ln_learned | 0.625 |
| pre_ln_rope | 0.625 |
| pre_ln_rmsnorm | 0.375 |
| pre_ln_swiglu | 0.625 |
| gqa_kv2 | 0.625 |

Regenerate: `python scripts/ablate.py`
