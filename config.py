import torch

config = {
    "vocab_size": 20002,
    "d_model": 128,
    "num_heads": 4,
    "d_ff": 512,
    "num_layers": 4,
    "num_classes": 2,
    "max_len": 512,
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 20,
    "dropout": 0.1,
    "warmup_steps": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}