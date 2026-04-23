# Model card: TinyShakespeare GPT (decoder-only)

## Intended use

Educational LM demonstrating a **decoder-only** Transformer with weight-tied embeddings, trained on the TinyShakespeare corpus.

## Training data

- [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (public domain–style Shakespeare texts bundled in many ML tutorials).
- Byte-level BPE tokenizer (default vocab size 2048).

## Metrics

Fill after training:

| Metric | Value |
|--------|------:|
| Val loss (nats) | _TBD_ |
| Perplexity | _TBD_ |

## Limitations

- Tiny corpus; model memorizes / fits training distribution; not a general language model.
- Generations are stylistic pastiche only.

## Ethical considerations

Generative models can produce offensive text if trained on unfiltered corpora. This demo corpus is small and domain-limited; still do not deploy without safety review.
