#!/usr/bin/env python3
"""Train IMDB sentiment classifier (Hydra)."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from transformer.data.imdb import get_imdb_dataloaders, save_vocab
from transformer.models.classifier import TransformerClassifier
from transformer.training.trainer import Trainer
from transformer.utils.artifacts import checkpoint_artifact, write_run_metadata


def _maybe_int_or_none(cfg: DictConfig, key: str) -> int | None:
    v = cfg.model.get(key)
    if v is None:
        return None
    return int(v)


@hydra.main(version_base=None, config_path="../configs", config_name="train_classifier")
def main(cfg: DictConfig) -> None:
    if str(cfg.train.device) == "auto":
        OmegaConf.update(cfg, "train.device", "cuda" if torch.cuda.is_available() else "cpu")

    out = Path(HydraConfig.get().runtime.output_dir)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    train_loader, val_loader, vocab = get_imdb_dataloaders(
        batch_size=int(cfg.data.batch_size),
        max_len=int(cfg.data.max_len),
        num_workers=int(cfg.data.num_workers),
        distributed=distributed,
    )
    vs = len(vocab)
    OmegaConf.update(cfg, "model.vocab_size", int(vs))
    vocab_path = save_vocab(vocab, out / "vocab.json")

    model = TransformerClassifier(
        vocab_size=int(vs),
        d_model=int(cfg.model.d_model),
        num_heads=int(cfg.model.num_heads),
        d_ff=int(cfg.model.d_ff),
        num_layers=int(cfg.model.num_layers),
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
        max_len=int(cfg.data.max_len),
        norm_first=bool(cfg.model.norm_first),
        pos_encoding=str(cfg.model.pos_encoding),
        use_rope=bool(cfg.model.use_rope),
        ffn_activation=str(cfg.model.ffn_activation),
        norm_type=str(cfg.model.get("norm_type", "layer_norm")),
        num_kv_heads=_maybe_int_or_none(cfg, "num_kv_heads"),
        alibi=bool(cfg.model.get("alibi", False)),
        use_checkpoint=bool(cfg.train.get("grad_ckpt", False)),
    )

    trainer = Trainer(cfg, "classifier", output_dir=out)
    summary = trainer.fit(model, train_loader, val_loader)
    if trainer.is_main_process:
        write_run_metadata(
            out,
            task="classifier",
            cfg=cfg,
            artifacts={
                "best_checkpoint": checkpoint_artifact(out / "best_model.pt"),
                "last_checkpoint": checkpoint_artifact(out / "last.pt"),
                "vocab": {"path": str(vocab_path), "format": "json"},
                "metrics_csv": {"path": str(out / str(cfg.train.csv_log)), "format": "csv"},
            },
            summary=summary,
        )


if __name__ == "__main__":
    main()
