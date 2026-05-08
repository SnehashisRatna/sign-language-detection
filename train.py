# train.py

import os
import sys
sys.path.append(os.path.abspath("."))   # Fix for Windows import issue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SignLanguageDataset
from models.gru_model import GRUClassifier
from scripts.utils import get_classes_from_data


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = "data"
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VAL_DIR         = os.path.join(DATA_DIR, "val")

CLASSES         = get_classes_from_data(TRAIN_DIR)

BATCH_SIZE      = 32
EPOCHS          = 100
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOP      = 15
CHECKPOINT_PATH = "gru_best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# DATASET (USING SPLIT FOLDERS)
# ─────────────────────────────────────────────

# Train → augmentation ON
train_dataset = SignLanguageDataset(
    TRAIN_DIR,
    CLASSES,
    augment_data=True,
    use_relative=True
)

# Validation → augmentation OFF
val_dataset = SignLanguageDataset(
    VAL_DIR,
    CLASSES,
    augment_data=False,
    use_relative=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n📊 Dataset loaded from split folders")
print(f"📂 Classes ({len(CLASSES)}): {CLASSES}")
print(f"   Train samples : {len(train_dataset)}")
print(f"   Val samples   : {len(val_dataset)}")
print(f"💻 Device        : {DEVICE}\n")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = GRUClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.4,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 Trainable parameters: {total_params:,}\n")


# Loss with label smoothing → better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# AdamW → better than Adam for generalization
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# Cosine LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)


# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
best_val_acc   = 0.0
patience_count = 0

print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>8} | {'LR':>9}")
print("─" * 58)

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ───────────────────────────────
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

        # Gradient clipping (important for GRU)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = 100 * train_correct / train_total

    # ── VALIDATION ──────────────────────────
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

    lr_now = scheduler.get_last_lr()[0]
    scheduler.step()

    print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_acc:>7.2f}% | {lr_now:>9.6f}")

    # ── SAVE BEST MODEL ─────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_count = 0

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_acc": val_acc,
            "classes": CLASSES,
        }, CHECKPOINT_PATH)

        print(f"         ✅ Saved best model (val_acc={val_acc:.2f}%)")

    else:
        patience_count += 1

        if patience_count >= EARLY_STOP:
            print(f"\n⏹ Early stopping — no improvement for {EARLY_STOP} epochs.")
            break


print("─" * 58)
print(f"\n🏆 Best Val Accuracy : {best_val_acc:.2f}%")
print(f"💾 Model saved at    : {CHECKPOINT_PATH}\n")