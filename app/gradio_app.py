"""Two-tab demo: IMDB sentiment (requires classifier ckpt) + GPT generation (requires GPT ckpt + tokenizer)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch

from transformer.data.tokenizers import load_tokenizer
from transformer.models.classifier import TransformerClassifier
from transformer.models.gpt import GPTModel


def load_clf(ckpt: Path, max_len: int = 256) -> tuple[TransformerClassifier, dict[str, int]]:
    # Minimal runtime vocab for demo: user should train with scripts/train_classifier.py
    # Here we use a tiny placeholder vocab if ckpt-only; for real use, ship vocab.json alongside ckpt.
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    words = "the a movie film good bad great terrible love hate amazing worst best".split()
    for i, w in enumerate(words, start=2):
        vocab[w] = i
    vs = max(vocab.values()) + 1
    m = TransformerClassifier(
        vocab_size=vs,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        num_classes=2,
        max_len=max_len,
    )
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    try:
        m.load_state_dict(state, strict=True)
    except Exception:
        m.load_state_dict(state, strict=False)
    m.eval()
    return m, vocab


def clf_infer(text: str, ckpt_path: str) -> str:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return "Checkpoint not found. Train with `python scripts/train_classifier.py` first."
    model, vocab = load_clf(ckpt)
    max_len = 256
    toks = [vocab.get(w, 1) for w in text.lower().split()][:max_len]
    toks = toks + [0] * (max_len - len(toks))
    x = torch.tensor([toks], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[0, 1].item()
    label = "positive" if p > 0.5 else "negative"
    return f"{label} (P(pos)={p:.3f}) — demo vocab is limited; train your own checkpoint for real IMDB performance."


def load_gpt(ckpt: Path, tok_path: Path) -> tuple[GPTModel, object]:
    tok = load_tokenizer(tok_path)
    vs = tok.get_vocab_size()
    m = GPTModel(
        vocab_size=vs,
        d_model=384,
        num_heads=6,
        d_ff=1536,
        num_layers=6,
        block_size=256,
        dropout=0.0,
        norm_first=True,
        use_rope=False,
    )
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    try:
        m.load_state_dict(state, strict=True)
    except Exception:
        m.load_state_dict(state, strict=False)
    m.eval()
    return m, tok


def gpt_gen(prompt: str, ckpt_path: str, tok_path: str, max_new: int, temperature: float) -> str:
    ckpt, tpath = Path(ckpt_path), Path(tok_path)
    if not ckpt.exists() or not tpath.exists():
        return "Missing checkpoint or tokenizer.json. Train GPT and ensure data/tinyshakespeare/tokenizer.json exists."
    model, tok = load_gpt(ckpt, tpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=int(max_new),
        temperature=float(temperature),
        top_k=50,
        use_kv_cache=True,
    )
    return tok.decode(out[0].tolist())


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Transformer from scratch — demo")
        with gr.Tab("Sentiment (encoder)"):
            t = gr.Textbox(label="Review text", lines=4)
            ck = gr.Textbox(label="Classifier checkpoint", value="best_model.pt")
            o = gr.Textbox(label="Prediction", lines=2)
            gr.Button("Classify").click(clf_infer, inputs=[t, ck], outputs=o)
        with gr.Tab("GPT generate"):
            p = gr.Textbox(label="Prompt", value="ROMEO:")
            ck2 = gr.Textbox(label="GPT checkpoint", value="best_model.pt")
            tk = gr.Textbox(label="tokenizer.json", value="data/tinyshakespeare/tokenizer.json")
            mn = gr.Slider(10, 400, value=120, step=10, label="max new tokens")
            temp = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="temperature")
            o2 = gr.Textbox(label="Output", lines=12)
            gr.Button("Generate").click(gpt_gen, inputs=[p, ck2, tk, mn, temp], outputs=o2)
    demo.launch()


if __name__ == "__main__":
    main()
