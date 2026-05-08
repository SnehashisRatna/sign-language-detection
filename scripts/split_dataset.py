import os
import shutil
import random

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = (".npz",)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# -------------------------
# HELPERS
# -------------------------
def get_classes():
    return [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and d not in ["train", "val", "test"]
    ]


def create_split_dirs(classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(DATA_DIR, split, cls)
            os.makedirs(path, exist_ok=True)


def get_files(cls_path):
    return [
        f for f in os.listdir(cls_path)
        if f.endswith(VALID_EXTENSIONS)
    ]


def move_files(files, src, dst):
    moved = 0
    for f in files:
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dst, f)

        if not os.path.exists(dst_path):  # prevent overwrite
            shutil.move(src_path, dst_path)
            moved += 1

    return moved


# -------------------------
# MAIN
# -------------------------
def split_dataset():
    print("\n🚀 Starting dataset split...\n")

    classes = get_classes()
    print(f"📊 Found {len(classes)} classes\n")

    create_split_dirs(classes)

    total_files = 0

    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        files = get_files(cls_path)

        if len(files) == 0:
            print(f"⚠️ Skipping {cls} (no files)")
            continue

        random.shuffle(files)

        n = len(files)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        train_dst = os.path.join(DATA_DIR, "train", cls)
        val_dst = os.path.join(DATA_DIR, "val", cls)
        test_dst = os.path.join(DATA_DIR, "test", cls)

        moved_train = move_files(train_files, cls_path, train_dst)
        moved_val = move_files(val_files, cls_path, val_dst)
        moved_test = move_files(test_files, cls_path, test_dst)

        total_files += moved_train + moved_val + moved_test

        print(
            f"✅ {cls}: "
            f"{moved_train} train | {moved_val} val | {moved_test} test"
        )

    print("\n🎯 Split completed successfully!")
    print(f"📦 Total files moved: {total_files}\n")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    split_dataset()