import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self._encode(self.texts[idx])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx])
    
    def _encode(self, text):
        words = text.lower().split()
        tokens = [self.vocab.get(w, 1) for w in words]
        tokens = tokens[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

def build_vocab(texts, vocab_size=20000):
    counts = Counter()
    for text in texts:
        counts.update(text.lower().split())
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(vocab_size))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab

def get_dataloaders(batch_size = 32, max_len=256):
    dataset = load_dataset("imdb")
    vocab = build_vocab(dataset["train"]["text"])

    train_ds = IMDBDataset(dataset["train"]["text"], dataset["train"]["label"],
                           vocab, max_len)
    test_ds = IMDBDataset(dataset["test"]["text"], dataset["test"]["label"],
                          vocab, max_len)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl, vocab