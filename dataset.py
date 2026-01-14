# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, classes):
        self.samples = []
        self.label_map = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in sorted(os.listdir(cls_dir)):   # 🔧 sorted
                if file.endswith(".npz"):
                    self.samples.append(
                        (os.path.join(cls_dir, file), self.label_map[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)["frames"]   # (30, 1662)

        # Optional safety check
        # assert data.shape == (30, 1662), f"Invalid shape: {data.shape}"

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
