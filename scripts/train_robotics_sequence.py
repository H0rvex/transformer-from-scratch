#!/usr/bin/env python3
"""Train a tiny GPT on synthetic discretized 2D trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transformer.data.robotics import TrajectorySpec, get_trajectory_dataloaders
from transformer.models.gpt import GPTModel
from transformer.training.metrics import lm_loss_and_perplexity
from transformer.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: GPTModel, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        logits = model(xb)
        assert isinstance(logits, torch.Tensor)
        loss, _ = lm_loss_and_perplexity(logits, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("outputs/robotics_sequence_smoke"))
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--num-sequences", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    spec = TrajectorySpec(
        grid_size=args.grid_size,
        block_size=args.block_size,
        num_sequences=args.num_sequences,
        seed=args.seed,
    )
    train_loader, val_loader = get_trajectory_dataloaders(spec, batch_size=args.batch_size)
    model = GPTModel(
        vocab_size=spec.vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        block_size=spec.block_size,
        dropout=0.1,
        norm_first=True,
        use_rope=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 1 if args.dry_run else args.epochs
    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            assert isinstance(logits, torch.Tensor)
            loss, _ = lm_loss_and_perplexity(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.dry_run:
                break

    val_loss = evaluate(model, val_loader)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": "synthetic_trajectory_next_token",
        "val_loss": val_loss,
        "grid_size": spec.grid_size,
        "block_size": spec.block_size,
        "num_sequences": spec.num_sequences,
        "dry_run": args.dry_run,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), args.out_dir / "best_model.pt")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
