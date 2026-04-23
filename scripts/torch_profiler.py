#!/usr/bin/env python3
"""Export a short Chrome trace for one LM training step (renamed to avoid shadowing stdlib `profile`)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from transformer.models.gpt import GPTModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/assets/trace.json"))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(
        vocab_size=args.vocab,
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_model * 4,
        num_layers=args.layers,
        block_size=args.seq,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    loss_fn = nn.CrossEntropyLoss()

    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with profile(activities=acts, record_shapes=True) as prof:
        with record_function("lm_step"):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.view(-1, args.vocab), y.view(-1))
            loss.backward()
            opt.step()
    prof.export_chrome_trace(str(args.out))
    print(f"Wrote trace to {args.out}")


if __name__ == "__main__":
    main()
