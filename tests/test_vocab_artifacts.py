from __future__ import annotations

from pathlib import Path

from transformer.data.imdb import encode_text, load_vocab, save_vocab


def test_vocab_save_load_roundtrip(tmp_path: Path) -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "good": 2}
    path = save_vocab(vocab, tmp_path / "vocab.json")
    assert load_vocab(path) == vocab
    assert encode_text("good missing", vocab, max_len=4) == [2, 1, 0, 0]
