# Transformer Sentiment Classifier

A Transformer encoder built **from scratch** in PyTorch, trained for binary sentiment classification on the [IMDB dataset](https://huggingface.co/datasets/imdb).

No `transformers` library — every component (multi-head attention, positional encoding, encoder layers) is implemented directly with `torch.nn`.

---

## Architecture

```
Input tokens
    └─ Token Embedding + Sinusoidal Positional Encoding
         └─ N × Encoder Layer
               ├─ Multi-Head Self-Attention  (with padding mask)
               ├─ Add & LayerNorm
               ├─ Feed-Forward Network  (Linear → ReLU → Linear)
               └─ Add & LayerNorm
         └─ Masked Mean Pooling  (ignores <pad> tokens)
              └─ Linear Classifier  → 2 logits (pos / neg)
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 128 |
| `num_heads` | 4 |
| `d_ff` | 512 |
| `num_layers` | 4 |
| `vocab_size` | 20,002 |
| `max_len` | 256 |
| `dropout` | 0.1 |
| `epochs` | 20 |
| `lr` | 1e-4 (cosine decay w/ warmup) |

---

## Project Structure

```
transformer/
├── model.py      # MultiHeadAttention, EncoderLayer, TransformerClassifier
├── data.py       # Vocabulary building, IMDB dataset & dataloaders
├── config.py     # All hyperparameters in one place
├── train.py      # Training loop, evaluation, model checkpointing
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads IMDB automatically via HuggingFace datasets)
python train.py
```

Training prints loss and accuracy after each epoch and saves the best checkpoint to `best_model.pt`.

---

## Key Implementation Details

- **Sinusoidal positional encoding** — fixed (not learned), following *Attention Is All You Need*.
- **Padding mask** — propagated through all attention layers so `<pad>` tokens never contribute to attention scores or the final pooled representation.
- **Masked mean pooling** — the sequence representation is the average of non-padding token embeddings, which is more stable than using the `[CLS]` position for a model trained from scratch.
- **Cosine LR schedule with linear warmup** — stabilises early training and improves convergence.
- **Gradient clipping** (`max_norm=1.0`) — prevents exploding gradients.

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Maas et al., [*Learning Word Vectors for Sentiment Analysis*](https://aclanthology.org/P11-1015/) — IMDB dataset
