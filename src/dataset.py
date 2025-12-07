from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import os

# Custom ImageFolder loader that skips corrupted images
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.samples = self._filter_corrupted(self.samples)

    def _filter_corrupted(self, samples):
        safe_samples = []
        for path, label in samples:
            try:
                with Image.open(path) as img:
                    img.verify()  # check if image is corrupted
                safe_samples.append((path, label))
            except (UnidentifiedImageError, OSError):
                print(f"Warning: skipping corrupted image {path}")
        return safe_samples

def get_dataloaders(data_dir, img_size=64, batch_size=32, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = SafeImageFolder(root=data_dir, transform=transform)
    
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, dataset.classes
