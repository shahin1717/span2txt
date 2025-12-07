import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError

def pil_loader_safe(path):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Warning: skipping corrupted image {path}")
        return None  # or handle differently


def get_dataloaders(data_dir, img_size=64, batch_size=32, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, loader=pil_loader_safe, transform=transform)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, dataset.classes
