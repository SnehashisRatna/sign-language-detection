"""
generate_gru_all_charts.py

This script generates the following report figures:

1. gru_classification_report.png
2. gru_training_curve.png
3. per_class_accuracy.png
4. gru_confusion_matrix.png
5. gru_roc_auc_curve.png
6. latency_throughput_chart.png
7. gru_lstm_comparison_chart.png

Expected optional input files:
- y_true.npy
- y_pred.npy
- y_prob.npy
- class_names.npy
- training_history.json

Run:
python generate_gru_all_charts.py
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 76

# If class_names.npy is not available, default class names will be used.
DEFAULT_CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def load_numpy_file(filename, default=None):
    if os.path.exists(filename):
        print(f"[LOADED] {filename}")
        return np.load(filename, allow_pickle=True)
    print(f"[MISSING] {filename} -> using default/sample data")
    return default


def load_class_names():
    if os.path.exists("class_names.npy"):
        class_names = np.load("class_names.npy", allow_pickle=True)
        return [str(c) for c in class_names]

    return DEFAULT_CLASS_NAMES


def create_sample_data(num_samples=1000, num_classes=76):
    """
    Creates sample data only if actual test output files are missing.
    Replace this by real y_true, y_pred, y_prob for final report.
    """
    np.random.seed(42)

    y_true = np.random.randint(0, num_classes, size=num_samples)

    y_pred = y_true.copy()

    # Add some errors
    error_indices = np.random.choice(num_samples, size=int(num_samples * 0.10), replace=False)
    y_pred[error_indices] = np.random.randint(0, num_classes, size=len(error_indices))

    y_prob = np.random.rand(num_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    # Increase probability for predicted class
    for i, pred in enumerate(y_pred):
        y_prob[i, pred] += 1.0

    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return y_true, y_pred, y_prob


# ============================================================
# 1. CLASSIFICATION REPORT IMAGE
# ============================================================

def plot_classification_report(y_true, y_pred, class_names):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []
    labels = []

    for cls in class_names:
        if cls in report_dict:
            rows.append([
                report_dict[cls]["precision"],
                report_dict[cls]["recall"],
                report_dict[cls]["f1-score"],
                report_dict[cls]["support"]
            ])
            labels.append(cls)

    # Add summary rows
    for avg_name in ["macro avg", "weighted avg"]:
        if avg_name in report_dict:
            rows.append([
                report_dict[avg_name]["precision"],
                report_dict[avg_name]["recall"],
                report_dict[avg_name]["f1-score"],
                report_dict[avg_name]["support"]
            ])
            labels.append(avg_name)

    rows = np.array(rows, dtype=object)

    fig_height = max(10, len(labels) * 0.28)
    plt.figure(figsize=(12, fig_height))
    plt.axis("off")

    table_data = []
    for row in rows:
        table_data.append([
            f"{row[0]:.3f}",
            f"{row[1]:.3f}",
            f"{row[2]:.3f}",
            f"{int(row[3])}"
        ])

    table = plt.table(
        cellText=table_data,
        rowLabels=labels,
        colLabels=["Precision", "Recall", "F1-Score", "Support"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    plt.title("GRU Classification Report", fontsize=16, pad=20)
    save_fig("gru_classification_report.png")


# ============================================================
# 2. TRAINING CURVE IMAGE
# ============================================================

def plot_training_curve():
    history = None

    if os.path.exists("training_history.json"):
        with open("training_history.json", "r") as f:
            history = json.load(f)
        print("[LOADED] training_history.json")
    else:
        print("[MISSING] training_history.json -> using sample training history")

        epochs = 50
        history = {
            "train_loss": list(np.linspace(2.5, 0.25, epochs) + np.random.normal(0, 0.05, epochs)),
            "val_loss": list(np.linspace(2.7, 0.45, epochs) + np.random.normal(0, 0.08, epochs)),
            "train_acc": list(np.linspace(0.35, 0.96, epochs) + np.random.normal(0, 0.01, epochs)),
            "val_acc": list(np.linspace(0.30, 0.91, epochs) + np.random.normal(0, 0.02, epochs))
        }

    train_loss = history.get("train_loss", history.get("train_losses", []))
    val_loss = history.get("val_loss", history.get("val_losses", []))
    train_acc = history.get("train_acc", history.get("train_accuracies", []))
    val_acc = history.get("val_acc", history.get("val_accuracies", []))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GRU Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("GRU Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    save_fig("gru_training_curve.png")


# ============================================================
# 3. PER-CLASS ACCURACY IMAGE
# ============================================================

def plot_per_class_accuracy(y_true, y_pred, class_names):
    per_class_acc = []

    for i in range(len(class_names)):
        indices = np.where(y_true == i)[0]

        if len(indices) == 0:
            acc = 0
        else:
            acc = np.mean(y_pred[indices] == y_true[indices])

        per_class_acc.append(acc)

    plt.figure(figsize=(18, 7))
    plt.bar(range(len(class_names)), per_class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy of GRU Model")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    save_fig("per_class_accuracy.png")


# ============================================================
# 4. CONFUSION MATRIX IMAGE
# ============================================================

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("GRU Confusion Matrix", fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    save_fig("gru_confusion_matrix.png")


# ============================================================
# 5. ROC-AUC CURVE IMAGE
# ============================================================

def plot_roc_auc_curve(y_true, y_prob, class_names):
    num_classes = len(class_names)

    try:
        y_true_bin = label_binarize(
            y_true,
            classes=list(range(num_classes))
        )

        fpr = {}
        tpr = {}
        roc_auc = {}

        plt.figure(figsize=(12, 9))

        # Plot only first 10 classes for readability
        max_plot_classes = min(10, num_classes)

        for i in range(max_plot_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                label=f"{class_names[i]} AUC = {roc_auc[i]:.2f}"
            )

        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        plt.plot(
            fpr_micro,
            tpr_micro,
            linestyle="--",
            linewidth=2,
            label=f"Micro-average AUC = {roc_auc_micro:.2f}"
        )

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("GRU ROC-AUC Curve")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True)

        save_fig("gru_roc_auc_curve.png")

    except Exception as e:
        print(f"[WARNING] Could not create ROC-AUC curve: {e}")

        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            "ROC-AUC curve could not be generated.\nCheck y_prob.npy shape.",
            ha="center",
            va="center",
            fontsize=14
        )
        plt.axis("off")
        save_fig("gru_roc_auc_curve.png")


# ============================================================
# 6. LATENCY AND THROUGHPUT CHART
# ============================================================

def plot_latency_throughput_chart():
    """
    Replace these values with your actual measured values if available.
    Example:
    GRU latency = 12 ms/frame
    LSTM latency = 18 ms/frame
    """
    models = ["GRU", "LSTM"]

    latency_ms = [12.0, 18.0]
    throughput_fps = [1000 / latency_ms[0], 1000 / latency_ms[1]]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(x - width / 2, latency_ms, width, label="Latency (ms/frame)")
    plt.bar(x + width / 2, throughput_fps, width, label="Throughput (FPS)")

    plt.xticks(x, models)
    plt.ylabel("Value")
    plt.title("Latency and Throughput Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    save_fig("latency_throughput_chart.png")


# ============================================================
# 7. GRU VS LSTM COMPARISON CHART
# ============================================================

def plot_gru_lstm_comparison_chart():
    """
    Replace these values after you train/test LSTM.

    Current old GRU values from your project:
    GRU old accuracy: 90.1%
    Macro F1: 0.903
    Weighted F1: 0.900

    You can update GRU new and LSTM values after final testing.
    """

    models = ["GRU Old", "GRU New", "LSTM"]

    accuracy = [0.901, 0.920, 0.890]
    macro_f1 = [0.903, 0.925, 0.885]
    weighted_f1 = [0.900, 0.922, 0.888]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, macro_f1, width, label="Macro F1")
    plt.bar(x + width, weighted_f1, width, label="Weighted F1")

    plt.xticks(x, models)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("GRU and LSTM Model Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    save_fig("gru_lstm_comparison_chart.png")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("\nGenerating GRU report figures...\n")

    class_names = load_class_names()
    num_classes = len(class_names)

    sample_y_true, sample_y_pred, sample_y_prob = create_sample_data(
        num_samples=1000,
        num_classes=num_classes
    )

    y_true = load_numpy_file("y_true.npy", sample_y_true)
    y_pred = load_numpy_file("y_pred.npy", sample_y_pred)
    y_prob = load_numpy_file("y_prob.npy", sample_y_prob)

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob)

    print(f"Number of classes: {num_classes}")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_prob shape: {y_prob.shape}")

    plot_classification_report(y_true, y_pred, class_names)
    plot_training_curve()
    plot_per_class_accuracy(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_auc_curve(y_true, y_prob, class_names)
    plot_latency_throughput_chart()
    plot_gru_lstm_comparison_chart()

    print("\nAll figures generated successfully.")
    print(f"Check folder: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()