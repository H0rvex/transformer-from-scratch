#!/usr/bin/env python3
"""Train GPT on TinyShakespeare (Hydra)."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from transformer.data.tinyshakespeare import ensure_tinyshakespeare_artifacts, get_tinyshakespeare_dataloaders
from transformer.models.gpt import GPTModel
from transformer.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_gpt")
def main(cfg: DictConfig) -> None:
    if str(cfg.train.device) == "auto":
        OmegaConf.update(cfg, "train.device", "cuda" if torch.cuda.is_available() else "cpu")

    out = Path(HydraConfig.get().runtime.output_dir)
    _, _, vs = ensure_tinyshakespeare_artifacts(
        Path(cfg.data.data_dir),
        vocab_size=int(cfg.data.vocab_bpe),
    )

    train_loader, val_loader, _ = get_tinyshakespeare_dataloaders(
        Path(cfg.data.data_dir),
        block_size=int(cfg.data.block_size),
        batch_size=int(cfg.data.batch_size),
        vocab_size=int(cfg.data.vocab_bpe),
        num_workers=int(cfg.data.num_workers),
    )

    model = GPTModel(
        vocab_size=int(vs),
        d_model=int(cfg.model.d_model),
        num_heads=int(cfg.model.num_heads),
        d_ff=int(cfg.model.d_ff),
        num_layers=int(cfg.model.num_layers),
        block_size=int(cfg.data.block_size),
        dropout=float(cfg.model.dropout),
        norm_first=bool(cfg.model.norm_first),
        use_rope=bool(cfg.model.use_rope),
    )

    trainer = Trainer(cfg, "lm", output_dir=out)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
