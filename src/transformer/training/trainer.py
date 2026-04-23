"""Shared training loop: classifier or LM with AMP, compile, logging, checkpoints."""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.training.callbacks import plot_loss_curve, save_classifier_plots
from transformer.training.metrics import classification_metrics, lm_loss_and_perplexity
from transformer.training.scheduler import get_cosine_schedule_with_warmup
from transformer.utils.logging_utils import CSVLogger, maybe_init_wandb, wandb_finish, wandb_log
from transformer.utils.seed import set_seed

Task = Literal["classifier", "lm"]


def configure_training_runtime_env() -> None:
    """Avoid HF tokenizer fork storms and OpenMP oversubscription (looks like 'many threads')."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _epoch_progress(loader: DataLoader[Any], desc: str) -> tqdm:
    """
    tqdm that behaves in Hydra / narrow terminals: stable width, slower refresh,
    disabled when stderr is not a TTY (avoids glued \\r lines in log capture).
    """
    total = len(loader)
    kwargs: dict[str, Any] = {
        "total": total,
        "mininterval": 0.5,
        "smoothing": 0.08,
        "file": sys.stderr,
        "dynamic_ncols": False,
    }
    try:
        kwargs["ncols"] = max(80, min(100, shutil.get_terminal_size(fallback=(100, 24)).columns))
    except OSError:
        kwargs["ncols"] = 100
    if not sys.stderr.isatty():
        kwargs["disable"] = True
    return tqdm(loader, desc=desc, **kwargs)


class Trainer:
    def __init__(self, cfg: DictConfig, task: Task, output_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.task = task
        self.output_dir = Path(output_dir or Path.cwd())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(str(cfg.train.device))
        self.grad_accum = int(cfg.train.grad_accum_steps)
        self.clip = float(cfg.train.clip_norm)
        self.amp_enabled = bool(cfg.train.amp)
        dtype_str = str(cfg.train.amp_dtype).lower()
        if dtype_str == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.use_scaler = self.amp_enabled and self.amp_dtype == torch.float16 and self.device.type == "cuda"

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> None:
        configure_training_runtime_env()
        set_seed(int(self.cfg.train.seed))
        model = model.to(self.device)
        if bool(self.cfg.train.compile) and not self.cfg.train.get("resume") and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

        if self.task == "classifier":
            self._fit_classifier(model, train_loader, val_loader)
        else:
            self._fit_lm(model, train_loader, val_loader)

    def _autocast(self) -> Any:
        """CUDA autocast only; `torch.cuda.amp.autocast` has no `device_type` (that is `torch.autocast`)."""
        if not self.amp_enabled or self.device.type != "cuda":
            return contextlib.nullcontext()
        dtype = self.amp_dtype
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)

    def _fit_classifier(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> None:
        cfg = self.cfg
        optim = AdamW(
            model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
        total_steps = max(1, int(cfg.train.epochs) * len(train_loader) // self.grad_accum)
        sched: LambdaLR = get_cosine_schedule_with_warmup(optim, int(cfg.train.warmup_steps), total_steps)
        scaler = GradScaler(enabled=self.use_scaler)
        loss_fn = nn.CrossEntropyLoss()
        csv_path = self.output_dir / str(cfg.train.csv_log)
        logger = CSVLogger(
            csv_path,
            ["epoch", "train_loss", "val_acc", "val_f1"],
        )
        _ = maybe_init_wandb(
            str(cfg.train.wandb_project),
            str(cfg.train.get("wandb_run_name") or "clf"),
            cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            bool(cfg.train.wandb),
        )

        start_epoch = 0
        global_step = 0
        ckpt_resume = cfg.train.get("resume")
        if ckpt_resume:
            start_epoch, global_step = self._load_checkpoint(Path(str(ckpt_resume)), model, optim, sched, scaler)

        best_acc = 0.0
        hist_ep: list[int] = []
        hist_tl: list[float] = []
        hist_va: list[float] = []

        for epoch in range(start_epoch, int(cfg.train.epochs)):
            model.train()
            running = 0.0
            n = 0
            optim.zero_grad(set_to_none=True)
            pbar = _epoch_progress(train_loader, desc=f"epoch {epoch + 1}/{cfg.train.epochs}")
            for step, (xb, yb) in enumerate(pbar):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                with self._autocast():
                    logits = model(xb)
                    loss = loss_fn(logits, yb) / self.grad_accum
                if self.use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.grad_accum == 0:
                    if self.use_scaler:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                    if self.use_scaler:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                running += float(loss.item()) * self.grad_accum
                n += 1
            avg_loss = running / max(1, n)
            val_m = self._eval_classifier(model, val_loader, loss_fn)
            acc = float(val_m["accuracy"])
            f1v = float(val_m["f1_macro"])
            logger.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_acc": acc, "val_f1": f1v})
            wandb_log({"train_loss": avg_loss, "val_acc": acc, "val_f1": f1v}, step=epoch + 1)
            hist_ep.append(epoch + 1)
            hist_tl.append(avg_loss)
            hist_va.append(acc)
            print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} val_acc={acc:.4f} val_f1={f1v:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
            self._save_checkpoint(self.output_dir / "last.pt", model, optim, sched, scaler, epoch + 1, global_step)

        save_classifier_plots(val_m, self.output_dir / "docs_assets")
        plot_loss_curve(hist_ep, hist_tl, hist_va, self.output_dir / "docs_assets" / "clf_curves.png", "val acc")
        wandb_finish()

    @torch.no_grad()
    def _eval_classifier(self, model: nn.Module, val_loader: DataLoader[Any], _loss_fn: nn.Module) -> dict[str, Any]:
        model.eval()
        ys: list[int] = []
        ps: list[int] = []
        scores: list[list[float]] = []
        for xb, yb in val_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with self._autocast():
                logits = model(xb)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            pred = logits.argmax(dim=-1).cpu().numpy()
            ys.extend(yb.cpu().numpy().tolist())
            ps.extend(pred.tolist())
            scores.extend(probs.tolist())
        return classification_metrics(np.array(ys), np.array(scores), np.array(ps))

    def _fit_lm(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> None:
        cfg = self.cfg
        optim = AdamW(
            model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
        total_steps = max(1, int(cfg.train.epochs) * len(train_loader) // self.grad_accum)
        sched = get_cosine_schedule_with_warmup(optim, int(cfg.train.warmup_steps), total_steps)
        scaler = GradScaler(enabled=self.use_scaler)

        csv_path = self.output_dir / str(cfg.train.csv_log)
        logger = CSVLogger(csv_path, ["epoch", "train_loss", "val_loss", "val_ppl"])
        _ = maybe_init_wandb(
            str(cfg.train.wandb_project),
            str(cfg.train.get("wandb_run_name") or "gpt"),
            cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            bool(cfg.train.wandb),
        )

        start_epoch = 0
        global_step = 0
        ckpt_resume = cfg.train.get("resume")
        if ckpt_resume:
            start_epoch, global_step = self._load_checkpoint(Path(str(ckpt_resume)), model, optim, sched, scaler)

        best_val = float("inf")
        hist_ep: list[int] = []
        hist_tl: list[float] = []
        hist_vl: list[float] = []

        for epoch in range(start_epoch, int(cfg.train.epochs)):
            model.train()
            running = 0.0
            n = 0
            optim.zero_grad(set_to_none=True)
            for step, (xb, yb) in enumerate(_epoch_progress(train_loader, desc=f"lm epoch {epoch + 1}")):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                with self._autocast():
                    logits = model(xb)
                    loss, _ppl = lm_loss_and_perplexity(logits, yb)
                    loss = loss / self.grad_accum
                if self.use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()  # type: ignore[no-untyped-call]

                if (step + 1) % self.grad_accum == 0:
                    if self.use_scaler:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                    if self.use_scaler:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                running += float(loss.item()) * self.grad_accum
                n += 1

            val_loss, val_ppl = self._eval_lm(model, val_loader)
            avg_loss = running / max(1, n)
            logger.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss, "val_ppl": val_ppl})
            wandb_log({"train_loss": avg_loss, "val_loss": val_loss, "val_ppl": val_ppl}, step=epoch + 1)
            hist_ep.append(epoch + 1)
            hist_tl.append(avg_loss)
            hist_vl.append(val_loss)
            print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
            self._save_checkpoint(self.output_dir / "last.pt", model, optim, sched, scaler, epoch + 1, global_step)

        plot_loss_curve(hist_ep, hist_tl, hist_vl, self.output_dir / "docs_assets" / "lm_curves.png", "val loss")
        wandb_finish()

    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        sched: LambdaLR,
        scaler: GradScaler,
        epoch: int,
        step: int,
    ) -> None:
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        if self.use_scaler:
            payload["scaler"] = scaler.state_dict()
        torch.save(payload, path)

    def _load_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        sched: LambdaLR,
        scaler: GradScaler,
    ) -> tuple[int, int]:
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location=self.device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None and self.use_scaler:
            scaler.load_state_dict(ck["scaler"])
        return int(ck.get("epoch", 0)), int(ck.get("step", 0))

    @torch.no_grad()
    def _eval_lm(self, model: nn.Module, val_loader: DataLoader[Any]) -> tuple[float, float]:
        model.eval()
        total = 0.0
        count = 0
        for xb, yb in val_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            with self._autocast():
                logits = model(xb)
                loss, ppl = lm_loss_and_perplexity(logits, yb)
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        mean_loss = total / max(1, count)
        return mean_loss, float(min(1e6, math.exp(mean_loss)))
