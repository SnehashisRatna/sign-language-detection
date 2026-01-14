# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SignLanguageDataset
from models.gru_model import GRUClassifier


# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data"

# ⚠️ MUST MATCH FOLDER NAMES + INFERENCE ORDER
CLASSES = [
    "AFTER", "ATTENTION", "BABY", "BEST", "BROTHER", "CHILD", "DAY",
    "DAYAFTER", "DAYAFTERTOMORROW", "DEAF", "DEGREE", "DELAY",
    "DO", "DONT", "DRINK", "family", "FATHER", "FOOD", "FRIEND",
    "HELLO", "HUNGER", "HUSBAND", "LATER", "MAN", "MILK", "MORE",
    "MOTHER", "NO", "NOW", "Numbers", "PLAY", "PLEASE", "PLENTY",
    "REGRET", "RICE", "ROTI", "SEND", "SISTER", "SORRY", "THANKYOU",
    "THERE", "THEY", "THIS", "THURSDAY", "TOMORROW", "WAIT",
    "WAHTEVER", "WALK", "WAKE", "WASTE", "WATER", "WAY", "WE",
    "WHAT", "WHERE", "WHICH", "WHILE", "WHO", "WHY", "WIDE",
    "WIFE", "WOMAN", "YEAR", "YES", "YESTERDAY"
]


BATCH_SIZE = 16          # ⬅ increased (important after augmentation)
EPOCHS = 50
LR = 1e-3
DEVICE = "cpu"


# -------------------------
# DATASET
# -------------------------
full_dataset = SignLanguageDataset(DATA_DIR, CLASSES)

# 80–20 split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True          # ✅ VERY IMPORTANT
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"📊 Dataset size: {len(full_dataset)}")
print(f"   Train: {len(train_dataset)}")
print(f"   Val  : {len(val_dataset)}")


# -------------------------
# MODEL
# -------------------------
model = GRUClassifier(
    input_size=1662,
    hidden_size=128,
    num_classes=len(CLASSES)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -------------------------
# TRAIN
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = 100 * train_correct / train_total

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )


# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "gru_sanity_model.pth")
print("✅ Model saved as gru_sanity_model.pth")
