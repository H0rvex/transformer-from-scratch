"""Decoder-only GPT with optional learned PE or RoPE, weight-tied LM head, KV cache."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from transformer.models.attention import RotaryEmbedding
from transformer.models.init_weights import init_embedding, init_linear, scaled_residual_linear
from transformer.models.layers import DecoderBlock
from transformer.models.norm import RMSNorm, build_norm
from transformer.models.positional import LearnedPositionalEmbedding


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_rope: bool = False,
        norm_type: str = "layer_norm",
        num_kv_heads: int | None = None,
        ffn_activation: str = "gelu",
        use_alibi: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        head_dim = d_model // num_heads
        rope = RotaryEmbedding(head_dim) if use_rope else None

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_rope:
            self.pos_emb: nn.Module = nn.Identity()
        else:
            self.pos_emb = LearnedPositionalEmbedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    norm_first=norm_first,
                    rope=rope,
                    ffn_activation=ffn_activation,
                    norm_type=norm_type,
                    num_kv_heads=num_kv_heads,
                    alibi=use_alibi,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = build_norm(norm_type, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights(num_layers)

    def _init_weights(self, n_layers: int) -> None:
        init_embedding(self.tok_emb)
        for block in self.blocks:
            assert isinstance(block, DecoderBlock)
            init_linear(block.attn.W_q)
            init_linear(block.attn.W_k)
            init_linear(block.attn.W_v)
            scaled_residual_linear(block.attn.W_o, n_layers)
            ff = block.ff
            if ff.activation == "swiglu":
                assert ff.w_gate is not None and ff.w_up is not None and ff.w_down is not None
                init_linear(ff.w_gate)
                init_linear(ff.w_up)
                scaled_residual_linear(ff.w_down, n_layers)
            else:
                assert ff.lin1 is not None and ff.lin2 is not None
                init_linear(ff.lin1)
                scaled_residual_linear(ff.lin2, n_layers)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(
        self,
        idx: Tensor,
        past_kv_layers: list[tuple[Tensor, Tensor] | None] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        _b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.block_size}")

        tok = self.tok_emb(idx)
        if isinstance(self.pos_emb, LearnedPositionalEmbedding):
            x = self.drop(self.pos_emb.forward_with_offset(tok, position_offset))
        elif isinstance(self.pos_emb, nn.Identity):
            x = self.drop(tok)
        else:
            x = self.drop(self.pos_emb(tok))

        layer_pasts = past_kv_layers if past_kv_layers is not None else [None] * len(self.blocks)
        presents: list[tuple[Tensor, Tensor]] = []

        for li, block in enumerate(self.blocks):
            past = layer_pasts[li]
            cache_len = past[0].size(2) if past is not None else 0
            out = block(
                x,
                attn_mask=None,
                past_kv=past,
                rope_position_offset=position_offset,
                cache_len=cache_len,
                use_cache=use_cache,
            )
            if use_cache and isinstance(out, tuple):
                x, present = out
                presents.append(present)
            else:
                assert isinstance(out, Tensor)
                x = out

        x = self.ln_f(x)
        logits = cast(Tensor, self.lm_head(x))
        if use_cache:
            return logits, presents
        return logits

    def _filter_logits(self, logits: Tensor, top_k: int | None, top_p: float | None) -> Tensor:
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        if top_p is not None and top_p > 0 and top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits.scatter_(-1, sorted_idx, sorted_logits)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_kv_cache: bool = True,
    ) -> Tensor:
        """Autoregressive sampling; uses KV cache when ``use_kv_cache``."""
        self.eval()
        block = self.block_size
        if max_new_tokens <= 0:
            return idx

        if not use_kv_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block:]
                logits = self(idx_cond)
                assert isinstance(logits, Tensor)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                logits = self._filter_logits(logits, top_k, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
            return idx

        raw = self(idx, use_cache=True, position_offset=0)
        assert isinstance(raw, tuple)
        logits, pasts = raw
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = self._filter_logits(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            pos_off = idx.size(1) - 1
            raw2 = self(
                idx[:, -1:],
                past_kv_layers=pasts,
                use_cache=True,
                position_offset=pos_off,
            )
            assert isinstance(raw2, tuple)
            logits, pasts = raw2
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = self._filter_logits(logits, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
