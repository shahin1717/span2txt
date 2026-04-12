import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm

from dataset import build_samples, split_samples, SignDataset, train_transforms, val_transforms
from model import build_model

# ── Config ──────────────────────────────────
DATA_DIR   = "../data_cleaned"
MODEL_PATH = "../sign_model.pth"
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_class_weights(train_samples, num_classes):
    counts  = Counter([label for _, label in train_samples])
    total   = len(train_samples)
    weights = [total / (num_classes * counts.get(i, 1))
               for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

def main():
    print(f"Device: {DEVICE}")

    samples, class_to_idx, classes = build_samples(DATA_DIR)
    train_s, val_s                 = split_samples(samples)
    num_classes                    = len(classes)

    print(f"Classes: {classes}")
    print(f"Train: {len(train_s)} | Val: {len(val_s)}")

    train_ds = SignDataset(train_s, train_transforms)
    val_ds   = SignDataset(val_s,   val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model(num_classes).to(DEVICE)
    weights   = compute_class_weights(train_s, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "class_to_idx": class_to_idx,
                "classes":      classes,
            }, MODEL_PATH)
            print(f"  ✓ saved (val_acc={val_acc:.3f})")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

if __name__ == "__main__":
    main()