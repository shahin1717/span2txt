import torch
import os

def save_checkpoint(model, optimizer, epoch, path="../experiments/checkpoints"):
    """
    Saves the model and optimizer state at the given epoch.
    """
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Loads the model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path} at epoch {epoch}")
    return epoch

def accuracy(outputs, labels):
    """
    Computes accuracy for predictions.
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)
