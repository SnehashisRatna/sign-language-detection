# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SignLanguageDataset
from models.gru_model import GRUClassifier


# CONFIG

DATA_DIR = "data"
CLASSES = ["HELLO", "SORRY", "THANKYOU"]
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-3
DEVICE = "cpu"


# DATA

dataset = SignLanguageDataset(DATA_DIR, CLASSES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# MODEL

model = GRUClassifier(
    input_size=1662,
    hidden_size=128,
    num_classes=len(CLASSES)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# TRAIN

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss:.4f}  Accuracy: {acc:.2f}%")


# SAVE MODEL

torch.save(model.state_dict(), "gru_sanity_model.pth")
print("✅ Model saved as gru_sanity_model.pth")
