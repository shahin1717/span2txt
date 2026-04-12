import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 128

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class SignDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples   # list of (image_path, label_index)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def build_samples(data_dir):
    classes      = sorted(os.listdir(data_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    samples      = []
    exts         = {".jpg", ".jpeg", ".png"}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in exts:
                samples.append((os.path.join(cls_dir, fname),
                                 class_to_idx[cls]))
    return samples, class_to_idx, classes

def split_samples(samples, val_split=0.2, seed=42):
    from collections import defaultdict
    import random
    random.seed(seed)

    by_class = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    train_s, val_s = [], []
    for cls_items in by_class.values():
        random.shuffle(cls_items)
        n_val = max(1, int(len(cls_items) * val_split))
        val_s.extend(cls_items[:n_val])
        train_s.extend(cls_items[n_val:])

    return train_s, val_s