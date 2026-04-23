from __future__ import annotations

from pathlib import Path

from transformer.data.tokenizers import CharTokenizer, train_bpe_tokenizer


def test_char_tokenizer_roundtrip() -> None:
    text = "abc\ndef"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_train_and_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "corpus.txt"
    p.write_text("hello world\nhello there\n", encoding="utf-8")
    out = tmp_path / "tok.json"
    tok = train_bpe_tokenizer([p], out, vocab_size=256)
    enc = tok.encode("hello world")
    dec = tok.decode(enc.ids)
    assert "hello" in dec and "world" in dec
