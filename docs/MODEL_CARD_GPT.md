# Model card: TinyShakespeare GPT (decoder-only)

## Intended use

Educational LM demonstrating a **decoder-only** Transformer with weight-tied embeddings, trained on the TinyShakespeare corpus.

## Training data

- [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (public domain–style Shakespeare texts bundled in many ML tutorials).
- Byte-level BPE tokenizer (default vocab size 2048).

## Metrics

All losses and perplexities are **per BPE token** (byte-level BPE, vocab 2048). They are **not** directly comparable to char-level reports (e.g. nanoGPT on TinyShakespeare).

Reference points for this architecture + corpus:

| Regime | Val loss (nats) | Val PPL |
|---|---:|---:|
| Uniform baseline (`ln 2048`) | 7.62 | 2048 |
| Unigram / frequency prior | ≈ 5.8–6.0 | ≈ 300–400 |
| **Target for this repo** | **4.0–5.0** | **55–150** |
| Stretch (risks overfit) | ~3.5 | ~30 |

Run-specific numbers (populated after a real run from `outputs/portfolio/gpt/metrics.csv`):

| Metric | Value |
|--------|------:|
| Val loss (nats) | _TBD_ |
| Perplexity | _TBD_ |

## Limitations

- Tiny corpus; model memorizes / fits training distribution; not a general language model.
- Generations are stylistic pastiche only.

## Ethical considerations

Generative models can produce offensive text if trained on unfiltered corpora. This demo corpus is small and domain-limited; still do not deploy without safety review.
