# scripts/class_distribution.py
import os

# =========================================================
# CONFIG
# =========================================================
DATA_DIR        = "data"
SPLITS          = ["train", "val", "test"]
WARN_THRESHOLD  = 50    # flag classes with total sequences below this


# =========================================================
# HELPERS
# =========================================================
def count_npz_files(folder_path):
    return len([
        f for f in os.listdir(folder_path)
        if f.endswith(".npz")
    ])


def get_all_classes():
    train_path = os.path.join(DATA_DIR, "train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Missing folder: {train_path}")
    return sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ])


# =========================================================
# MAIN
# =========================================================
def main():
    classes = get_all_classes()

    print("\n" + "=" * 95)
    print("  📊 DATASET CLASS DISTRIBUTION")
    print("=" * 95)

    header = (
        f"  {'':3}"
        f"{'CLASS':<20}"
        f"{'TRAIN':>10}"
        f"{'VAL':>10}"
        f"{'TEST':>10}"
        f"{'TOTAL':>12}"
        f"{'RATIO':>10}"
    )
    print(header)
    print("  " + "-" * 90)

    grand_train = 0
    grand_val   = 0
    grand_test  = 0
    warned      = []

    for cls in classes:
        counts = {}
        for split in SPLITS:
            split_path = os.path.join(DATA_DIR, split, cls)
            counts[split] = (
                count_npz_files(split_path)
                if os.path.exists(split_path) else 0
            )

        total = counts["train"] + counts["val"] + counts["test"]

        grand_train += counts["train"]
        grand_val   += counts["val"]
        grand_test  += counts["test"]

        # Low sample warning flag
        flag = "⚠️ " if total < WARN_THRESHOLD else "✅ "
        if total < WARN_THRESHOLD:
            warned.append((cls, total))

        # Train/Val/Test ratio display
        if total > 0:
            ratio = (
                f"{counts['train']/total*100:.0f}/"
                f"{counts['val']/total*100:.0f}/"
                f"{counts['test']/total*100:.0f}"
            )
        else:
            ratio = "N/A"

        print(
            f"  {flag}"
            f"{cls:<20}"
            f"{counts['train']:>10}"
            f"{counts['val']:>10}"
            f"{counts['test']:>10}"
            f"{total:>12}"
            f"{ratio:>10}"
        )

    # ── Grand Total ───────────────────────────────────────
    grand_total = grand_train + grand_val + grand_test
    grand_ratio = (
        f"{grand_train/grand_total*100:.0f}/"
        f"{grand_val/grand_total*100:.0f}/"
        f"{grand_test/grand_total*100:.0f}"
        if grand_total > 0 else "N/A"
    )

    print("  " + "-" * 90)
    print(
        f"  {'   '}"
        f"{'TOTAL':<20}"
        f"{grand_train:>10}"
        f"{grand_val:>10}"
        f"{grand_test:>10}"
        f"{grand_total:>12}"
        f"{grand_ratio:>10}"
    )
    print("=" * 95)

    # ── Summary Stats ─────────────────────────────────────
    avg = grand_total / len(classes) if classes else 0
    print(f"\n  📦 Total sequences  : {grand_total}")
    print(f"  🏷️  Total classes    : {len(classes)}")
    print(f"  📈 Avg per class    : {avg:.1f}")
    print(f"  🔀 Split ratio      : {grand_ratio} (train/val/test)")

    if warned:
        print(f"\n  ⚠️  Low-sample classes (< {WARN_THRESHOLD} sequences):")
        for cls, total in warned:
            print(f"     → {cls}: {total} sequences — consider adding more data")
    else:
        print(f"\n  ✅ All classes have ≥ {WARN_THRESHOLD} sequences")

    print("\n" + "=" * 95 + "\n")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()