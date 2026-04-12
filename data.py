from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self._encode(self.texts[idx])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx])

    def _encode(self, text: str):
        words = text.lower().split()
        tokens = [self.vocab.get(w, 1) for w in words]  # 1 = <unk>
        tokens = tokens[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))    # 0 = <pad>
        return tokens


def build_vocab(texts, vocab_size: int = 20000) -> dict:
    counts = Counter()
    for text in texts:
        counts.update(text.lower().split())
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(vocab_size))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def get_dataloaders(batch_size: int = 32, max_len: int = 256):
    dataset = load_dataset("imdb")
    vocab = build_vocab(dataset["train"]["text"])

    train_ds = IMDBDataset(dataset["train"]["text"], dataset["train"]["label"], vocab, max_len)
    test_ds = IMDBDataset(dataset["test"]["text"], dataset["test"]["label"], vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader, vocab
