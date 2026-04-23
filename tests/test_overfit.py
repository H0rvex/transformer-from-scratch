from __future__ import annotations

import torch
import torch.nn as nn

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def test_classifier_overfits_two_samples() -> None:
    torch.manual_seed(0)
    model = TransformerClassifier(
        vocab_size=32,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
        max_len=8,
        norm_first=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 0, 0]])
    y = torch.tensor([0, 1])
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    assert (pred == y).all()


def test_gpt_overfits_two_samples() -> None:
    torch.manual_seed(1)
    block = 8
    vs = 40
    model = GPTModel(
        vocab_size=vs,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        block_size=block,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randint(1, vs - 1, (2, block))
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits[:, :-1].contiguous().view(-1, vs), x[:, 1:].contiguous().view(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(x)
        pred = logits[:, :-1].argmax(dim=-1)
    assert (pred == x[:, 1:]).all()
