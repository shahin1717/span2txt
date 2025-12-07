import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import BATCH_SIZE, LR, EPOCHS, DATA_DIR
from model import CNNModel
from dataset import get_dataloaders
from utils import save_checkpoint, load_checkpoint, accuracy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloaders
train_loader, test_loader, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

# Model
model = CNNModel(num_classes=len(classes)).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop with tqdm for ETA
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    
    for batch_idx, (images, labels) in enumerate(loop, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        avg_loss = running_loss / batch_idx
        
        # Update tqdm bar with dynamic info
        loop.set_postfix(loss=f"{avg_loss:.4f}")
    
    print(f"Epoch {epoch+1}/{EPOCHS} completed, Avg Loss: {avg_loss:.4f}")
    
    # Optional: save checkpoint every epoch
    save_checkpoint(model, optimizer, epoch+1)
