#!/usr/bin/env python3
"""Train IMDB sentiment classifier (Hydra)."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from transformer.data.imdb import get_imdb_dataloaders
from transformer.models.classifier import TransformerClassifier
from transformer.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_classifier")
def main(cfg: DictConfig) -> None:
    if str(cfg.train.device) == "auto":
        OmegaConf.update(cfg, "train.device", "cuda" if torch.cuda.is_available() else "cpu")

    out = Path(HydraConfig.get().runtime.output_dir)
    train_loader, val_loader, vocab = get_imdb_dataloaders(
        batch_size=int(cfg.data.batch_size),
        max_len=int(cfg.data.max_len),
        num_workers=int(cfg.data.num_workers),
    )
    vs = len(vocab)

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
    )

    trainer = Trainer(cfg, "classifier", output_dir=out)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
