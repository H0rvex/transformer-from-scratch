#!/usr/bin/env python3
"""Save per-head attention heatmaps using the first encoder block's attention only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from transformer.models.layers import EncoderBlock


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default="the movie was great and not bad")
    p.add_argument("--out-dir", type=Path, default=Path("docs/assets/attention"))
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    args = p.parse_args()

    words = args.text.lower().split()
    stoi: dict[str, int] = {}
    for w in words:
        if w not in stoi:
            stoi[w] = len(stoi) + 2
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    vs = max(stoi.values()) + 1
    ids = [stoi.get(w, 1) for w in words]
    max_len = 32
    ids = ids + [0] * (max_len - len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    mask = (x != 0).unsqueeze(1).unsqueeze(2)

    block = EncoderBlock(
        args.d_model,
        args.heads,
        args.d_ff,
        dropout=0.0,
        norm_first=False,
        rope=None,
        ffn_activation="relu",
    )
    emb = nn.Embedding(vs, args.d_model, padding_idx=0)
    with torch.no_grad():
        h = emb(x)
        out = block.attn(h, attn_mask=mask, is_causal=False, return_attn_weights=True)
        assert isinstance(out, tuple)
        _, w = out
        w0 = w[0].cpu().numpy()
    t = len(words)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_heads = w0.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 3), squeeze=False)
    im = None
    for hi in range(n_heads):
        ax = axes[0, hi]
        im = ax.imshow(w0[hi, :t, :t], cmap="magma", vmin=0.0, vmax=max(1e-6, float(w0.max())))
        ax.set_title(f"head {hi}")
        ax.set_xlabel("key")
        ax.set_ylabel("query")
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.suptitle(args.text)
    fig.tight_layout()
    out_path = args.out_dir / "layer0_heads.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
