"""BPE (byte-level) and character tokenizers."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    corpus_paths: list[str | Path],
    out_path: Path | str,
    vocab_size: int = 2048,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train([str(Path(p).resolve()) for p in corpus_paths], trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: Path | str) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_file_to_memmap(
    tokenizer: Tokenizer,
    corpus_path: Path | str,
    out_memmap_path: Path | str,
) -> int:
    text = Path(corpus_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    arr_path = Path(out_memmap_path)
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    arr = np.array(ids, dtype=np.uint32)
    np.save(str(arr_path), arr)
    return int(arr.max()) + 1  # rough vocab usage


class CharTokenizer:
    """Character-level tokenizer for small corpora."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
