import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from model import TransformerClassifier
import data 
import math
from config import config

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))
    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    train_loader, test_loader, vocab = data.get_dataloaders()

    model = TransformerClassifier(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    ).to(config["device"])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = config["epochs"] * len(train_loader)
    scheduler = get_scheduler(optimizer, warmup_steps=config["warmup_steps"], total_steps=total_steps)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(config["device"]), labels.to(config["device"])
            # forward
            logits = model(texts)
            # loss
            loss = loss_fn(logits, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(config["device"]), labels.to(config["device"])
                preds = model(texts).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1}/{config['epochs']} | Avg Loss: {total_loss/len(train_loader):.4f} | Accuracy: {correct/total * 100:.4f}")