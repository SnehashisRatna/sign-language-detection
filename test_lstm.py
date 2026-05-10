# test_lstm.py
"""
LSTM Test Evaluation + GRU vs LSTM Comparison Table
=====================================================
Run AFTER test.py (GRU) and train_lstm.py.
Generates:
  1. LSTM test metrics
  2. Side-by-side GRU vs LSTM comparison chart
  3. Comparison table saved to file
Run: python test_lstm.py
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
from models.lstm_model import LSTMClassifier
from scripts.utils import get_classes_from_data


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_DIR       = "data/test"
LSTM_MODEL     = "lstm_best_model.pth"
GRU_REPORT     = "test_results/test_classification_report.txt"   # from test.py
BATCH_SIZE     = 32
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR       = "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD LSTM MODEL
# ─────────────────────────────────────────────
print("\n" + "═"*58)
print("   LSTM TEST EVALUATION  —  GRU vs LSTM Comparison")
print("═"*58)

print("\n📦 Loading LSTM model...")
ckpt    = torch.load(LSTM_MODEL, map_location=DEVICE)
CLASSES = ckpt.get("classes", get_classes_from_data(TEST_DIR))

model = LSTMClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.0,              # ✅ OFF during evaluation
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()

lstm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ LSTM loaded — {len(CLASSES)} classes")
print(f"   Best Val Acc (training): {ckpt.get('val_acc', 0):.2f}%")
print(f"   Parameters             : {lstm_params:,}")


# ─────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────
print("\n📂 Loading test data...")
test_ds = SignLanguageDataset(TEST_DIR, CLASSES,
                               augment_data=False, use_relative=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
print(f"✅ Test samples: {len(test_ds)}\n")


# ─────────────────────────────────────────────
# INFERENCE + LATENCY
# ─────────────────────────────────────────────
print("🔍 Running LSTM predictions...")
all_preds = []; all_labels = []; all_probs = []; latencies = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)
        t0      = time.perf_counter()
        outputs = model(X)
        t1      = time.perf_counter()

        latencies.extend([(t1-t0)/X.size(0)*1000] * X.size(0))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)


# ─────────────────────────────────────────────
# LSTM METRICS
# ─────────────────────────────────────────────
lstm_acc = accuracy_score(all_labels, all_preds) * 100
lstm_p, lstm_r, lstm_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0)
lstm_p *= 100; lstm_r *= 100; lstm_f1 *= 100

try:
    y_bin    = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    lstm_auc = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro') * 100
    lstm_auc_str = f"{lstm_auc:.2f}%"
except:
    lstm_auc = 0.0
    lstm_auc_str = "N/A"

lstm_lat = np.mean(latencies)
lstm_fps = 1000.0 / lstm_lat

print(f"\n{'─'*58}")
print(f"   LSTM TEST RESULTS")
print(f"{'─'*58}")
print(f"   Accuracy        : {lstm_acc:.2f}%")
print(f"   Macro Precision : {lstm_p:.2f}%")
print(f"   Macro Recall    : {lstm_r:.2f}%")
print(f"   Macro F1-Score  : {lstm_f1:.2f}%")
print(f"   ROC-AUC (macro) : {lstm_auc_str}")
print(f"   Avg Latency     : {lstm_lat:.2f} ms  ({lstm_fps:.1f} FPS)")
print(f"{'─'*58}\n")


# ─────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────
report = classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=3, zero_division=0)
print(report)
rp = os.path.join(SAVE_DIR, "lstm_classification_report.txt")
with open(rp, "w", encoding="utf-8") as f:
    f.write("LSTM — TEST SET EVALUATION\n" + "="*55 + "\n\n")
    f.write(f"Accuracy  : {lstm_acc:.2f}%\nF1-Score  : {lstm_f1:.2f}%\n\n")
    f.write(report)
print(f"✅ Saved: {rp}")


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(28, 24))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.3, ax=ax, annot_kws={"size": 7})
ax.set_title(f"Confusion Matrix — LSTM TEST SET\nAccuracy: {lstm_acc:.2f}%  |  F1: {lstm_f1:.2f}%",
             fontsize=15, fontweight="bold", pad=20)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lstm_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: lstm_confusion_matrix.png")


# ─────────────────────────────────────────────
# GRU vs LSTM COMPARISON CHART
# ─────────────────────────────────────────────
print("\n📊 Generating GRU vs LSTM Comparison Chart...")

# ── Read GRU results from test_results/test_classification_report.txt
gru_acc = gru_f1 = gru_auc = gru_lat = gru_fps = gru_params = None

if os.path.exists(GRU_REPORT):
    with open(GRU_REPORT, "r") as f:
        for line in f:
            if "Accuracy"        in line: gru_acc = float(line.split(":")[1].strip().replace("%",""))
            if "Macro F1-Score"  in line: gru_f1  = float(line.split(":")[1].strip().replace("%",""))
            if "ROC-AUC"         in line:
                try: gru_auc = float(line.split(":")[1].strip().replace("%",""))
                except: gru_auc = 0.0
            if "Avg Latency"     in line:
                try: gru_lat = float(line.split(":")[1].strip().split()[0])
                except: gru_lat = 0.0
            if "Throughput"      in line:
                try: gru_fps = float(line.split(":")[1].strip().split()[0])
                except: gru_fps = 0.0
else:
    print("⚠️  GRU report not found. Using placeholder values.")
    print("   Run python test.py first to get GRU results.\n")
    gru_acc = gru_f1 = gru_auc = 0.0
    gru_lat = gru_fps = 0.0

# GRU parameter count (from checkpoint)
try:
    from models.gru_model import GRUClassifier
    gru_model_tmp = GRUClassifier(1662, 256, len(CLASSES), 0.0)
    gru_params    = sum(p.numel() for p in gru_model_tmp.parameters() if p.requires_grad)
    del gru_model_tmp
except:
    gru_params = 0

# ── Bar chart comparison
metrics_labels = ["Accuracy (%)", "F1-Score (%)", "ROC-AUC (%)"]
gru_vals  = [gru_acc  or 0, gru_f1  or 0, gru_auc  or 0]
lstm_vals = [lstm_acc,       lstm_f1,       lstm_auc or 0]

x     = np.arange(len(metrics_labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("GRU vs LSTM — Comparison\nReal-Time Indian Sign Language Detection",
             fontsize=15, fontweight="bold")

# Accuracy / F1 / AUC
ax = axes[0]
b1 = ax.bar(x - width/2, gru_vals,  width, label="GRU (Proposed)", color="#2ecc71", edgecolor="white")
b2 = ax.bar(x + width/2, lstm_vals, width, label="LSTM (Baseline)", color="#3498db", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=11); ax.set_ylim(0, 115)
ax.set_title("Accuracy / F1 / AUC", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1,
            f"{h:.1f}", ha="center", fontsize=9, fontweight="bold")

# Parameters + Latency
ax2 = axes[1]
comp_labels = ["Parameters (K)", "Latency (ms)", "FPS"]
gru_c  = [(gru_params or 0)/1000, gru_lat or 0, gru_fps or 0]
lstm_c = [lstm_params/1000,        lstm_lat,      lstm_fps]
x2     = np.arange(len(comp_labels))
c1 = ax2.bar(x2 - width/2, gru_c,  width, label="GRU",  color="#2ecc71", edgecolor="white")
c2 = ax2.bar(x2 + width/2, lstm_c, width, label="LSTM", color="#3498db", edgecolor="white")
ax2.set_xticks(x2); ax2.set_xticklabels(comp_labels, fontsize=11)
ax2.set_title("Efficiency Comparison", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, axis="y")
for bar in list(c1) + list(c2):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3,
             f"{h:.1f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
comp_path = os.path.join(SAVE_DIR, "gru_vs_lstm_comparison.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {comp_path}")


# ─────────────────────────────────────────────
# COMPARISON TABLE (saved to file)
# ─────────────────────────────────────────────
table_path = os.path.join(SAVE_DIR, "gru_vs_lstm_table.txt")
with open(table_path, "w") as f:
    f.write("GRU vs LSTM — COMPARISON TABLE\n")
    f.write("="*65 + "\n\n")
    f.write(f"{'Metric':<25} {'GRU (Proposed)':>18} {'LSTM (Baseline)':>18}\n")
    f.write("-"*65 + "\n")
    f.write(f"{'Test Accuracy':<25} {str(gru_acc or 0)+' %':>18} {lstm_acc:.2f} %\n".replace("0 %","N/A"))
    f.write(f"{'Macro F1-Score':<25} {str(gru_f1 or 0)+' %':>18} {lstm_f1:.2f} %\n")
    f.write(f"{'ROC-AUC (macro)':<25} {str(gru_auc or 0)+' %':>18} {lstm_auc_str:>18}\n")
    f.write(f"{'Parameters':<25} {str((gru_params or 0)):>18} {lstm_params:>18}\n")
    f.write(f"{'Avg Latency (ms)':<25} {str(gru_lat or 0):>18} {lstm_lat:.2f}\n")
    f.write(f"{'Throughput (FPS)':<25} {str(gru_fps or 0):>18} {lstm_fps:.1f}\n")
    f.write("="*65 + "\n")

with open(table_path, "r") as f:
    print("\n" + f.read())
print(f"✅ Saved: {table_path}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*58}")
print(f"   GRU vs LSTM COMPARISON COMPLETE")
print(f"{'═'*58}")
print(f"   Saved to: {SAVE_DIR}/")
print(f"   ├── lstm_classification_report.txt")
print(f"   ├── lstm_confusion_matrix.png")
print(f"   ├── gru_vs_lstm_comparison.png  ← USE IN REPORT")
print(f"   └── gru_vs_lstm_table.txt       ← USE IN REPORT")
print(f"{'═'*58}\n")