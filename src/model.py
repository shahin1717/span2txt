import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    # Load MobileNetV2 pretrained on ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks so they adapt to our data
    for layer in model.features[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier head: 1280 → 512 → num_classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model