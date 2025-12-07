# src/evaluate.py
import torch
from .dataset import get_dataloaders
from .model import CNNModel
from .config import DATA_DIR, BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

# Load the latest checkpoint
model = CNNModel(num_classes=len(classes)).to(device)
checkpoint_path = "../experiments/checkpoints/checkpoint_epoch_20.pt"  # adjust to last epoch
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")
