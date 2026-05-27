# scripts/split_dataset.py

import os
import shutil
import random

# =========================================================
# CONFIG
# =========================================================

# MASTER DATASET (all .npz files)
SOURCE_DIR = "merged_dataset"

# OUTPUT SPLITS
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

VALID_EXTENSIONS = (".npz",)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =========================================================
# HELPERS
# =========================================================

def get_classes():
    return sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])


def create_split_dirs(classes):

    # create output root
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:

        split_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)


def get_files(class_path):

    return sorted([
        f for f in os.listdir(class_path)
        if f.endswith(VALID_EXTENSIONS)
    ])


def copy_files(files, src_dir, dst_dir):

    copied = 0

    for f in files:

        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)

        # avoid overwrite
        if not os.path.exists(dst_path):

            shutil.copy2(src_path, dst_path)
            copied += 1

    return copied


# =========================================================
# MAIN SPLIT FUNCTION
# =========================================================

def split_dataset():

    print("\n🚀 Starting dataset split...\n")

    classes = get_classes()

    print(f"📊 Found {len(classes)} classes\n")

    create_split_dirs(classes)

    total_files = 0

    for cls in classes:

        cls_path = os.path.join(SOURCE_DIR, cls)

        files = get_files(cls_path)

        if len(files) == 0:
            print(f"⚠️ Skipping {cls} (no files)")
            continue

        random.shuffle(files)

        n = len(files)

        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_files = files[:train_end]
        val_files   = files[train_end:val_end]
        test_files  = files[val_end:]

        # destination folders
        train_dst = os.path.join(OUTPUT_DIR, "train", cls)
        val_dst   = os.path.join(OUTPUT_DIR, "val", cls)
        test_dst  = os.path.join(OUTPUT_DIR, "test", cls)

        copied_train = copy_files(train_files, cls_path, train_dst)
        copied_val   = copy_files(val_files, cls_path, val_dst)
        copied_test  = copy_files(test_files, cls_path, test_dst)

        total_files += copied_train + copied_val + copied_test

        print(
            f"✅ {cls}: "
            f"{copied_train} train | "
            f"{copied_val} val | "
            f"{copied_test} test"
        )

    print("\n🎯 Split completed successfully!")
    print(f"📦 Total files copied: {total_files}\n")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    split_dataset()