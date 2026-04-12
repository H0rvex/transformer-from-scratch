import torch

config = {
    # Vocabulary
    "vocab_size": 20002,      # 20 000 words + <pad> (0) + <unk> (1)

    # Model architecture
    "d_model": 128,           # Embedding / hidden dimension
    "num_heads": 4,           # Attention heads (d_model must be divisible)
    "d_ff": 512,              # Feed-forward inner dimension
    "num_layers": 4,          # Number of encoder layers
    "num_classes": 2,         # Positive / Negative
    "dropout": 0.1,

    # Data
    "max_len": 256,           # Max token sequence length
    "batch_size": 32,

    # Training
    "lr": 1e-4,
    "epochs": 20,
    "warmup_steps": 500,      # Linear LR warmup before cosine decay

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
