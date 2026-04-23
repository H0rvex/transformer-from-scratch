from __future__ import annotations

import torch
import torch.nn as nn

from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def test_classifier_loss_three_steps_deterministic() -> None:
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
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randint(1, 30, (4, 8))
    y = torch.randint(0, 2, (4,))

    losses: list[float] = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()

    torch.manual_seed(0)
    model2 = TransformerClassifier(
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
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    losses2: list[float] = []
    for _ in range(3):
        opt2.zero_grad(set_to_none=True)
        logits = model2(x)
        loss = loss_fn(logits, y)
        losses2.append(float(loss.item()))
        loss.backward()
        opt2.step()

    assert losses == losses2


def test_gpt_loss_three_steps_deterministic() -> None:
    torch.manual_seed(1)
    vs = 40
    block = 8
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

    def run() -> list[float]:
        losses: list[float] = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits[:, :-1].reshape(-1, vs), x[:, 1:].reshape(-1))
            losses.append(float(loss.item()))
            loss.backward()
            opt.step()
        return losses

    l1 = run()
    torch.manual_seed(1)
    model2 = GPTModel(
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
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-2)

    def run2() -> list[float]:
        losses: list[float] = []
        for _ in range(3):
            opt2.zero_grad(set_to_none=True)
            logits = model2(x)
            loss = loss_fn(logits[:, :-1].reshape(-1, vs), x[:, 1:].reshape(-1))
            losses.append(float(loss.item()))
            loss.backward()
            opt2.step()
        return losses

    assert l1 == run2()
