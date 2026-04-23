#!/usr/bin/env python3
"""Run quick architecture ablations on synthetic data and write docs/ABLATIONS.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from transformer.models.classifier import TransformerClassifier
from transformer.training.trainer import Trainer


def _tiny_cfg(epochs: int, lr: float, norm_first: bool, pos: str, use_rope: bool) -> object:
    return OmegaConf.create(
        {
            "train": {
                "epochs": epochs,
                "lr": lr,
                "warmup_steps": 5,
                "weight_decay": 0.0,
                "grad_accum_steps": 1,
                "clip_norm": 1.0,
                "amp": False,
                "amp_dtype": "fp32",
                "compile": False,
                "wandb": False,
                "wandb_project": "ablate",
                "wandb_run_name": None,
                "csv_log": "metrics.csv",
                "seed": 0,
                "device": "cpu",
                "resume": None,
            },
            "model": {
                "d_model": 32,
                "num_heads": 4,
                "d_ff": 64,
                "num_layers": 2,
                "num_classes": 2,
                "dropout": 0.0,
                "max_len": 16,
                "norm_first": norm_first,
                "pos_encoding": pos,
                "use_rope": use_rope,
                "ffn_activation": "relu",
            },
            "data": {"batch_size": 4, "max_len": 16, "num_workers": 0},
        }
    )


def _synthetic_loaders() -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    torch.manual_seed(0)
    x = torch.randint(1, 50, (8, 16))
    y = torch.randint(0, 2, (8,))
    ds = torch.utils.data.TensorDataset(x, y)
    return ds, ds


def run_case(name: str, cfg: object, out_dir: Path) -> float:
    train_ds, val_ds = _synthetic_loaders()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
    model = TransformerClassifier(
        vocab_size=60,
        d_model=int(cfg.model.d_model),
        num_heads=int(cfg.model.num_heads),
        d_ff=int(cfg.model.d_ff),
        num_layers=int(cfg.model.num_layers),
        num_classes=2,
        dropout=float(cfg.model.dropout),
        max_len=int(cfg.model.max_len),
        norm_first=bool(cfg.model.norm_first),
        pos_encoding=str(cfg.model.pos_encoding),
        use_rope=bool(cfg.model.use_rope),
        ffn_activation=str(cfg.model.ffn_activation),
    )
    t = Trainer(cfg, "classifier", output_dir=out_dir / name)
    t.fit(model, train_loader, val_loader)
    # read last val acc from printed? simpler: eval here
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum())
            total += int(yb.numel())
    return correct / max(1, total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/ABLATIONS.md"))
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    base = Path("outputs/ablate_runs")
    base.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float]] = []

    cases = [
        ("post_ln_sinusoidal", _tiny_cfg(args.epochs, 3e-3, False, "sinusoidal", False)),
        ("pre_ln_sinusoidal", _tiny_cfg(args.epochs, 3e-3, True, "sinusoidal", False)),
        ("pre_ln_learned", _tiny_cfg(args.epochs, 3e-3, True, "learned", False)),
        ("pre_ln_rope", _tiny_cfg(args.epochs, 3e-3, True, "none", True)),
    ]
    for name, cfg in cases:
        acc = run_case(name, cfg, base)
        rows.append((name, acc))

    lines = [
        "# Ablations (synthetic sanity)",
        "",
        "Short runs on a fixed synthetic binary dataset (CPU) to compare **pre-LN vs post-LN**,",
        "**positional encoding variants**, and **RoPE** at tiny width.",
        "",
        "| setting | val accuracy |",
        "|---|---:|",
    ]
    for n, a in rows:
        lines.append(f"| {n} | {a:.3f} |")
    lines.append("")
    lines.append("Regenerate: `python scripts/ablate.py`")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
