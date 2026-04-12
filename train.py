import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import data
from config import config
from model import TransformerClassifier


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        logits = model(texts)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = config["device"]
    print(f"Using device: {device}\n")

    train_loader, test_loader, _ = data.get_dataloaders(
        batch_size=config["batch_size"], max_len=config["max_len"]
    )

    model = TransformerClassifier(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = config["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config["warmup_steps"], total_steps
    )

    best_acc = 0.0
    for epoch in range(1, config["epochs"] + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nBest test accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
