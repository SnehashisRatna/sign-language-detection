# scripts/evaluate.py
"""
Complete Model Evaluation Script
Generates:
  1. Confusion Matrix
  2. Training History Graph (loss + accuracy)
  3. Per-class Precision, Recall, F1 Score
  4. Overall Classification Report
"""

import os
import sys
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split

from models.gru_model import GRUClassifier
from dataset import SignLanguageDataset
from scripts.utils import get_classes_from_data


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "gru_best_model.pth"
DATA_DIR    = "data"
BATCH_SIZE  = 32
DEVICE      = "cpu"
OUTPUT_DIR  = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("\n📦 Loading model...")
CLASSES = get_classes_from_data(DATA_DIR)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    CLASSES    = ckpt.get("classes", CLASSES)
    state_dict = ckpt["model_state"]
    history    = ckpt.get("history", None)
else:
    state_dict = ckpt
    history    = None

model = GRUClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.0,
).to(DEVICE)

model.load_state_dict(state_dict)
model.eval()
print(f"✅ Model loaded — {len(CLASSES)} classes")


# ─────────────────────────────────────────────
# LOAD VALIDATION DATA
# ─────────────────────────────────────────────
print("📂 Loading validation data...")

full_ds    = SignLanguageDataset(DATA_DIR, CLASSES, augment_data=False, use_relative=True)
total      = len(full_ds)
val_size   = int(0.2 * total)
train_size = total - val_size

_, val_split = random_split(
    full_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✅ Validation samples: {len(val_split)}")


# ─────────────────────────────────────────────
# RUN PREDICTIONS
# ─────────────────────────────────────────────
print("🔍 Running predictions...")

all_preds  = []
all_labels = []

with torch.no_grad():
    for X, y in val_loader:
        X      = X.to(DEVICE)
        output = model(X)
        preds  = torch.argmax(output, dim=1).cpu().numpy()
        labels = y.numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = (all_preds == all_labels).mean() * 100
print(f"✅ Overall Accuracy: {overall_acc:.2f}%\n")


# ─────────────────────────────────────────────
# 1. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("─" * 60)
print("  CLASSIFICATION REPORT (Precision / Recall / F1)")
print("─" * 60)

report = classification_report(
    all_labels, all_preds,
    target_names=CLASSES,
    digits=3,
    zero_division=0
)
print(report)

# Save to file
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Overall Accuracy: {overall_acc:.2f}%\n\n")
    f.write(report)
print(f"✅ Saved: {OUTPUT_DIR}/classification_report.txt")


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n📊 Generating Confusion Matrix...")

cm = confusion_matrix(all_labels, all_preds)

# Large figure for 76 classes
fig, ax = plt.subplots(figsize=(28, 24))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    linewidths=0.3,
    linecolor="gray",
    ax=ax,
    annot_kws={"size": 7}
)

ax.set_title(
    f"Confusion Matrix — Sign Language Detection\nOverall Accuracy: {overall_acc:.2f}%",
    fontsize=16, fontweight="bold", pad=20
)
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("True Label",      fontsize=12, labelpad=10)

plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {cm_path}")


# ─────────────────────────────────────────────
# 3. PER-CLASS ACCURACY BAR CHART
# ─────────────────────────────────────────────
print("\n📊 Generating Per-Class Accuracy Chart...")

per_class_acc = []
for i in range(len(CLASSES)):
    mask     = all_labels == i
    if mask.sum() == 0:
        per_class_acc.append(0.0)
        continue
    correct  = (all_preds[mask] == i).sum()
    acc      = correct / mask.sum() * 100
    per_class_acc.append(acc)

# Color bars: green ≥ 80%, orange 50-80%, red < 50%
colors = []
for acc in per_class_acc:
    if acc >= 80:
        colors.append("#2ecc71")
    elif acc >= 50:
        colors.append("#f39c12")
    else:
        colors.append("#e74c3c")

fig, ax = plt.subplots(figsize=(22, 8))
bars = ax.bar(CLASSES, per_class_acc, color=colors, edgecolor="white", linewidth=0.5)

ax.set_title(
    "Per-Class Accuracy\n🟢 ≥80%   🟠 50–80%   🔴 <50%",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Sign Class", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.axhline(y=80, color="green",  linestyle="--", linewidth=1, alpha=0.6, label="80% threshold")
ax.axhline(y=50, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")

plt.xticks(rotation=90, fontsize=7)
plt.legend(fontsize=10)
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {bar_path}")


# ─────────────────────────────────────────────
# 4. TRAINING HISTORY (Loss + Accuracy)
# ─────────────────────────────────────────────
if history:
    print("\n📊 Generating Training History...")

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss",   [])
    train_acc  = history.get("train_acc",  [])
    val_acc    = history.get("val_acc",    [])
    epochs     = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Loss curve
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss",      linewidth=2)
    ax1.plot(epochs, val_loss,   "r-o", markersize=3, label="Validation Loss", linewidth=2)
    ax1.set_title("Loss over Epochs",     fontsize=13)
    ax1.set_xlabel("Epoch",               fontsize=11)
    ax1.set_ylabel("Loss",                fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, train_acc, "b-o", markersize=3, label="Train Accuracy",      linewidth=2)
    ax2.plot(epochs, val_acc,   "r-o", markersize=3, label="Validation Accuracy", linewidth=2)
    ax2.set_title("Accuracy over Epochs", fontsize=13)
    ax2.set_xlabel("Epoch",               fontsize=11)
    ax2.set_ylabel("Accuracy (%)",        fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {hist_path}")

else:
    print("⚠️  Training history not found in checkpoint.")
    print("   To save history, add it to your train.py checkpoint save.")


# ─────────────────────────────────────────────
# 5. FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  EVALUATION COMPLETE")
print(f"{'═'*55}")
print(f"  Overall Accuracy : {overall_acc:.2f}%")
print(f"  Classes          : {len(CLASSES)}")
print(f"  Val Samples      : {len(val_split)}")
print(f"\n  Files saved to: {OUTPUT_DIR}/")
print(f"  ├── confusion_matrix.png")
print(f"  ├── per_class_accuracy.png")
print(f"  ├── training_history.png  (if history saved)")
print(f"  └── classification_report.txt")
print(f"{'═'*55}\n")