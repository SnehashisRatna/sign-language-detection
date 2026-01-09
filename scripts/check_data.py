# scripts/check_data.py
import os
import numpy as np

DATA_DIR = "../data"

total = 0

for label in sorted(os.listdir(DATA_DIR)):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    files = [f for f in os.listdir(label_dir) if f.endswith(".npz")]
    print(f"\nClass: {label} | Samples: {len(files)}")

    for f in files[:3]:
        path = os.path.join(label_dir, f)
        d = np.load(path)
        print(f"  {f} -> frames: {d['frames'].shape}, label in file: {d['label']}")
        total += 1

print(f"\nTotal samples checked: {total}")
