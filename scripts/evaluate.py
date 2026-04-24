#!/usr/bin/env python3
"""Evaluate saved classifier or GPT checkpoints from metadata/config artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from transformer.data.imdb import get_imdb_dataloaders, load_vocab
from transformer.data.tinyshakespeare import get_tinyshakespeare_dataloaders
from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel
from transformer.training.metrics import classification_metrics, lm_loss_and_perplexity
from transformer.utils.artifacts import load_run_metadata


def _cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.metadata is not None:
        return dict(load_run_metadata(args.metadata)["config"])
    if args.config is not None:
        raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        assert isinstance(raw, dict)
        return raw
    raise SystemExit("Provide --metadata or --config.")


def _state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state dict")
    return model_state


def _artifact_path(args: argparse.Namespace, key: str) -> Path | None:
    if args.metadata is None:
        return None
    meta = load_run_metadata(args.metadata)
    artifact = meta.get("artifacts", {}).get(key)
    if isinstance(artifact, dict) and artifact.get("path"):
        return Path(str(artifact["path"]))
    return None


def _classifier_from_cfg(cfg: dict[str, Any], vocab_size: int) -> TransformerClassifier:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    return TransformerClassifier(
        vocab_size=vocab_size,
        d_model=int(mcfg["d_model"]),
        num_heads=int(mcfg["num_heads"]),
        d_ff=int(mcfg["d_ff"]),
        num_layers=int(mcfg["num_layers"]),
        num_classes=int(mcfg["num_classes"]),
        dropout=0.0,
        max_len=int(dcfg.get("max_len", mcfg.get("max_len", 256))),
        norm_first=bool(mcfg.get("norm_first", False)),
        pos_encoding=str(mcfg.get("pos_encoding", "sinusoidal")),
        use_rope=bool(mcfg.get("use_rope", False)),
        ffn_activation=str(mcfg.get("ffn_activation", "relu")),
        norm_type=str(mcfg.get("norm_type", "layer_norm")),
        num_kv_heads=mcfg.get("num_kv_heads"),
        alibi=bool(mcfg.get("alibi", False)),
    )


def _gpt_from_cfg(cfg: dict[str, Any]) -> GPTModel:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    return GPTModel(
        vocab_size=int(mcfg["vocab_size"]),
        d_model=int(mcfg["d_model"]),
        num_heads=int(mcfg["num_heads"]),
        d_ff=int(mcfg["d_ff"]),
        num_layers=int(mcfg["num_layers"]),
        block_size=int(dcfg.get("block_size", mcfg.get("block_size", 256))),
        dropout=0.0,
        norm_first=bool(mcfg.get("norm_first", True)),
        use_rope=bool(mcfg.get("use_rope", False)),
        norm_type=str(mcfg.get("norm_type", "layer_norm")),
        num_kv_heads=mcfg.get("num_kv_heads"),
        ffn_activation=str(mcfg.get("ffn_activation", "gelu")),
        use_alibi=bool(mcfg.get("use_alibi", False)),
    )


def _synthetic_classifier_loader(max_len: int) -> tuple[DataLoader[Any], dict[str, int]]:
    vocab = {"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3}
    x = torch.tensor(
        [
            [2, 2, 0, 0],
            [3, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    if max_len > x.size(1):
        pad = torch.zeros(x.size(0), max_len - x.size(1), dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
    x = x[:, :max_len]
    y = torch.tensor([1, 0], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2), vocab


def evaluate_classifier(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _cfg_from_args(args)
    device = _device(args.device)
    max_len = int(cfg["data"].get("max_len", cfg["model"].get("max_len", 256)))
    vocab_path = args.vocab or _artifact_path(args, "vocab")
    if args.synthetic_smoke:
        loader, vocab = _synthetic_classifier_loader(max_len)
    else:
        if vocab_path is None:
            raise SystemExit("Classifier evaluation needs --vocab or metadata with artifacts.vocab.")
        vocab = load_vocab(vocab_path)
        _, loader, _ = get_imdb_dataloaders(
            batch_size=int(cfg["data"].get("batch_size", 32)),
            max_len=max_len,
            num_workers=int(cfg["data"].get("num_workers", 0)),
        )

    model = _classifier_from_cfg(cfg, len(vocab)).to(device)
    model.load_state_dict(_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    ys: list[int] = []
    ps: list[int] = []
    scores: list[list[float]] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            pred = logits.argmax(dim=-1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ps.extend(pred.tolist())
            scores.extend(probs.tolist())
    out = classification_metrics(np.array(ys), np.array(scores), np.array(ps))
    out["task"] = "classifier"
    return out


def _synthetic_gpt_loader(block_size: int, vocab_size: int) -> DataLoader[Any]:
    base = torch.arange(block_size + 1).unsqueeze(0).repeat(4, 1) % vocab_size
    return DataLoader(TensorDataset(base[:, :-1].long(), base[:, 1:].long()), batch_size=2)


def evaluate_gpt(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _cfg_from_args(args)
    device = _device(args.device)
    model = _gpt_from_cfg(cfg).to(device)
    model.load_state_dict(_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    block_size = int(cfg["data"].get("block_size", cfg["model"].get("block_size", 256)))
    if args.synthetic_smoke:
        loader = _synthetic_gpt_loader(block_size, int(cfg["model"]["vocab_size"]))
    else:
        _, loader, _ = get_tinyshakespeare_dataloaders(
            Path(str(cfg["data"]["data_dir"])),
            block_size=block_size,
            batch_size=int(cfg["data"].get("batch_size", 32)),
            vocab_size=int(cfg["data"].get("vocab_bpe", cfg["model"]["vocab_size"])),
            num_workers=int(cfg["data"].get("num_workers", 0)),
        )

    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            assert isinstance(logits, torch.Tensor)
            loss, _ = lm_loss_and_perplexity(logits, yb)
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    loss = total / max(1, n)
    return {"task": "gpt", "val_loss": loss, "val_ppl": float(np.exp(min(20.0, loss)))}


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("classifier", "gpt"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--vocab", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--out", type=Path, default=Path("eval_metrics.json"))
    args = p.parse_args()

    metrics = evaluate_classifier(args) if args.task == "classifier" else evaluate_gpt(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
