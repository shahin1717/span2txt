import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_samples, SignDataset, val_transforms
from model import build_model

DATA_DIR   = "../data_cleaned"
MODEL_PATH = "../model/sign_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ckpt        = torch.load(MODEL_PATH, map_location=DEVICE)
    classes     = ckpt["classes"]
    num_classes = len(classes)

    model = build_model(num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples, _, _ = build_samples(DATA_DIR)
    dataset       = SignDataset(samples, val_transforms)
    loader        = DataLoader(dataset, batch_size=32,
                               shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Overall accuracy: {acc*100:.2f}%\n")
    print(classification_report(all_labels, all_preds,
                                target_names=classes, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes,
                yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — acc={acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved → confusion_matrix.png")

if __name__ == "__main__":
    main()