# test.py
"""
Research-Grade Test Evaluation
================================
Run: python test.py
Outputs saved to: test_results/
"""

import os
import sys
import time
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dataset import SignLanguageDataset
from models.gru_model import GRUClassifier
from scripts.utils import get_classes_from_data


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_DIR    = "data/test"
MODEL_PATH  = "gru_best_model.pth"
BATCH_SIZE  = 32
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD MODEL + CLASSES FROM CHECKPOINT
# ─────────────────────────────────────────────
print("\n" + "═"*55)
print("   SIGN LANGUAGE — TEST SET EVALUATION")
print("═"*55)

print("\n📦 Loading model...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# ✅ Load classes from checkpoint — guaranteed to match training
CLASSES = ckpt.get("classes", get_classes_from_data(TEST_DIR))

model = GRUClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.0,          # ✅ MUST be 0.0 during evaluation
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()             # ✅ Sets BatchNorm to eval mode too

print(f"✅ Model loaded — {len(CLASSES)} classes")
print(f"   Best Val Acc (training): {ckpt.get('val_acc', 0):.2f}%")
print(f"💻 Device: {DEVICE}")


# ─────────────────────────────────────────────
# LOAD TEST DATASET
# ─────────────────────────────────────────────
print("\n📂 Loading test data...")
test_dataset = SignLanguageDataset(
    TEST_DIR,
    CLASSES,
    augment_data=False,    # ✅ Never augment test data
    use_relative=True      # ✅ Must match training
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
print(f"✅ Test samples: {len(test_dataset)}\n")


# ─────────────────────────────────────────────
# INFERENCE + LATENCY MEASUREMENT
# ─────────────────────────────────────────────
print("🔍 Running predictions...")

all_preds  = []
all_labels = []
all_probs  = []
latencies  = []           # ms per sample

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)

        t0      = time.perf_counter()
        outputs = model(X)
        t1      = time.perf_counter()

        # ms per sample in this batch
        batch_lat = (t1 - t0) / X.size(0) * 1000
        latencies.extend([batch_lat] * X.size(0))

        probs  = torch.softmax(outputs, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
accuracy = accuracy_score(all_labels, all_preds)

# ✅ macro average — treats all classes equally (correct for research)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds,
    average="macro",
    zero_division=0
)

# ✅ ROC-AUC (One-vs-Rest, macro)
try:
    y_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    auc   = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
    auc_str = f"{auc * 100:.2f}%"
except Exception as e:
    auc_str = "N/A"
    print(f"   AUC skipped: {e}")

# Latency
avg_lat = np.mean(latencies)
fps     = 1000.0 / avg_lat


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"   TEST RESULTS  (Held-Out Test Set)")
print(f"{'─'*55}")
print(f"   Accuracy        : {accuracy * 100:.2f}%")
print(f"   Macro Precision : {precision * 100:.2f}%")
print(f"   Macro Recall    : {recall * 100:.2f}%")
print(f"   Macro F1-Score  : {f1 * 100:.2f}%")
print(f"   ROC-AUC (macro) : {auc_str}")
print(f"   Avg Latency     : {avg_lat:.2f} ms/sample")
print(f"   Throughput      : {fps:.1f} FPS")
print(f"{'─'*55}\n")


# ─────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────
report = classification_report(
    all_labels, all_preds,
    target_names=CLASSES,
    digits=3,
    zero_division=0
)
print(report)

report_path = os.path.join(SAVE_DIR, "test_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("SIGN LANGUAGE DETECTION — TEST SET EVALUATION\n")
    f.write("="*55 + "\n\n")
    f.write(f"Accuracy        : {accuracy * 100:.2f}%\n")
    f.write(f"Macro Precision : {precision * 100:.2f}%\n")
    f.write(f"Macro Recall    : {recall * 100:.2f}%\n")
    f.write(f"Macro F1-Score  : {f1 * 100:.2f}%\n")
    f.write(f"ROC-AUC (macro) : {auc_str}\n")
    f.write(f"Avg Latency     : {avg_lat:.2f} ms/sample\n")
    f.write(f"Throughput      : {fps:.1f} FPS\n\n")
    f.write(report)
print(f"✅ Saved: {report_path}")


# ─────────────────────────────────────────────
# CONFUSION MATRIX  (seaborn heatmap)
# ─────────────────────────────────────────────
print("\n📊 Generating Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(28, 24))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES,
    linewidths=0.3, linecolor="gray",
    ax=ax, annot_kws={"size": 7}
)
ax.set_title(
    f"Confusion Matrix — TEST SET\n"
    f"Accuracy: {accuracy*100:.2f}%  |  F1: {f1*100:.2f}%  |  AUC: {auc_str}",
    fontsize=15, fontweight="bold", pad=20
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "test_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {cm_path}")


# ─────────────────────────────────────────────
# PER-CLASS ACCURACY  (color coded)
# ─────────────────────────────────────────────
print("\n📊 Generating Per-Class Accuracy Chart...")
per_class_acc = []
for i in range(len(CLASSES)):
    mask = all_labels == i
    if mask.sum() == 0:
        per_class_acc.append(0.0)
        continue
    per_class_acc.append((all_preds[mask] == i).mean() * 100)

# ✅ Color coded: green=good, orange=average, red=weak
colors = [
    "#2ecc71" if a >= 80 else
    "#f39c12" if a >= 50 else
    "#e74c3c"
    for a in per_class_acc
]

fig, ax = plt.subplots(figsize=(22, 8))
ax.bar(CLASSES, per_class_acc, color=colors, edgecolor="white", linewidth=0.5)
ax.set_title(
    "Per-Class Accuracy — TEST SET\n"
    "Green >= 80%   Orange = 50-80%   Red < 50%",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Sign Class", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.axhline(y=80, color="green",  linestyle="--", linewidth=1, alpha=0.6)
ax.axhline(y=50, color="orange", linestyle="--", linewidth=1, alpha=0.6)
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
bar_path = os.path.join(SAVE_DIR, "test_per_class_accuracy.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {bar_path}")


# ─────────────────────────────────────────────
# LATENCY ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 Generating Latency Chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Inference Latency Analysis", fontsize=14, fontweight="bold")

ax1.hist(latencies, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
ax1.axvline(avg_lat, color="red", linestyle="--", linewidth=2,
            label=f"Mean: {avg_lat:.2f} ms")
ax1.set_title("Latency Distribution")
ax1.set_xlabel("ms / sample")
ax1.set_ylabel("Count")
ax1.legend()
ax1.grid(True, alpha=0.3)

labels2 = ["Avg (ms)", "Min (ms)", "Max (ms)", "FPS"]
vals2   = [avg_lat, np.min(latencies), np.max(latencies), fps]
clrs2   = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
bars    = ax2.bar(labels2, vals2, color=clrs2, edgecolor="white")
for bar, val in zip(bars, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Latency Summary")
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
lat_path = os.path.join(SAVE_DIR, "test_latency_analysis.png")
plt.savefig(lat_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {lat_path}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"   EVALUATION COMPLETE")
print(f"{'═'*55}")
print(f"   Accuracy        : {accuracy * 100:.2f}%")
print(f"   Macro Precision : {precision * 100:.2f}%")
print(f"   Macro Recall    : {recall * 100:.2f}%")
print(f"   Macro F1-Score  : {f1 * 100:.2f}%")
print(f"   ROC-AUC (macro) : {auc_str}")
print(f"   Avg Latency     : {avg_lat:.2f} ms  ({fps:.1f} FPS)")
print(f"\n   Saved to: {SAVE_DIR}/")
print(f"   ├── test_classification_report.txt")
print(f"   ├── test_confusion_matrix.png")
print(f"   ├── test_per_class_accuracy.png")
print(f"   └── test_latency_analysis.png")
print(f"{'═'*55}\n")