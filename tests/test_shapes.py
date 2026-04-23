from __future__ import annotations

import torch

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def test_classifier_shapes() -> None:
    m = TransformerClassifier(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=3,
        max_len=24,
    ).eval()
    x = torch.randint(0, 100, (5, 24))
    y = m(x)
    assert y.shape == (5, 3)


def test_gpt_shapes() -> None:
    m = GPTModel(
        vocab_size=120,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        block_size=16,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    ).eval()
    x = torch.randint(0, 120, (4, 10))
    logits = m(x)
    assert logits.shape == (4, 10, 120)
