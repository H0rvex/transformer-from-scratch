#!/usr/bin/env python3
"""Sample text from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer.data.tokenizers import load_tokenizer
from transformer.models.gpt import GPTModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("best_model.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tinyshakespeare/tokenizer.json"))
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--no-kv-cache", action="store_true", help="Disable KV-cache (slower reference)")
    p.add_argument("--out", type=Path, default=Path("docs/assets/generations.md"))
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer)
    vs = tok.get_vocab_size()
    model = GPTModel(
        vocab_size=int(vs),
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_ff,
        num_layers=args.layers,
        block_size=args.block_size,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    ).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)

    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=not args.no_kv_cache,
    )
    text = tok.decode(out_ids[0].tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# Sample\n\n**Prompt:** {args.prompt}\n\n{text}\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
