# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from scripts.utils import to_relative_holistic, FACE_FEAT, HAND_N


# ── Feature slice indices (within 1662-dim vector) ────────────────────────────
# Face:       468 × 3 = 1404   → [0    : 1404]
# Pose:        33 × 4 = 132    → [1404 : 1536]
# Left hand:   21 × 3 = 63     → [1536 : 1599]
# Right hand:  21 × 3 = 63     → [1599 : 1662]
LH_START = FACE_FEAT + 33 * 4
RH_START = LH_START + HAND_N * 3


# ── Augmentation functions ────────────────────────────────────────────────────

def _gaussian_noise(seq, std=0.005):
    """Add small Gaussian noise — simulates sensor jitter."""
    return seq + np.random.normal(0, std, seq.shape).astype(np.float32)


def _time_warp(seq, factor_range=(0.85, 1.15)):
    """Stretch/compress sequence temporally — simulates signing speed variation."""
    T, F = seq.shape
    factor   = np.random.uniform(*factor_range)
    new_len  = max(2, int(T * factor))
    orig_t   = np.linspace(0, 1, T)
    warp_t   = np.linspace(0, 1, new_len)
    out      = np.zeros((T, F), dtype=np.float32)
    for i in range(F):
        warped = interp1d(orig_t, seq[:, i])(np.clip(warp_t, 0, 1))
        out[:, i] = interp1d(warp_t, warped)(orig_t)
    return out


def _temporal_crop(seq, crop_ratio=0.10):
    """Randomly trim start/end frames — simulates late/early capture."""
    T       = len(seq)
    max_c   = max(1, int(T * crop_ratio))
    start   = np.random.randint(0, max_c + 1)
    end     = T - np.random.randint(0, max_c + 1)
    cropped = seq[start:end]
    pad_f   = np.repeat(cropped[:1],  start,   axis=0)
    pad_b   = np.repeat(cropped[-1:], T - end, axis=0)
    return np.concatenate([pad_f, cropped, pad_b], axis=0)[:T]


def _mirror_hands(seq):
    """
    Swap left ↔ right hand landmarks and flip x-coordinates.
    Also mirrors face and pose x-coords for global consistency.
    Useful: makes model hand-agnostic.
    """
    s = seq.copy()

    # Mirror face x (every 3rd element starting at 0, within face block)
    s[:, 0:FACE_FEAT:3]  = 1.0 - s[:, 0:FACE_FEAT:3]

    # Mirror pose x (every 4th element starting at 1404)
    s[:, FACE_FEAT:FACE_FEAT + 33*4:4] = 1.0 - s[:, FACE_FEAT:FACE_FEAT + 33*4:4]

    # Swap left and right hand blocks
    lh = s[:, LH_START:LH_START + HAND_N*3].copy()
    rh = s[:, RH_START:RH_START + HAND_N*3].copy()
    s[:, LH_START:LH_START + HAND_N*3] = rh
    s[:, RH_START:RH_START + HAND_N*3] = lh

    # Mirror hand x-coords after swap
    s[:, LH_START:LH_START + HAND_N*3:3] = 1.0 - s[:, LH_START:LH_START + HAND_N*3:3]
    s[:, RH_START:RH_START + HAND_N*3:3] = 1.0 - s[:, RH_START:RH_START + HAND_N*3:3]

    return s


def _scale(seq, scale_range=(0.92, 1.08)):
    """Randomly scale all coordinates — simulates distance from camera."""
    return seq * np.random.uniform(*scale_range)


def augment(seq):
    """
    Apply a random combination of augmentations to a (30, 1662) sequence.
    Each augmentation is applied independently with its own probability.
    """
    if np.random.random() < 0.5:
        seq = _gaussian_noise(seq)
    if np.random.random() < 0.4:
        seq = _time_warp(seq)
    if np.random.random() < 0.3:
        seq = _temporal_crop(seq)
    if np.random.random() < 0.4:
        seq = _mirror_hands(seq)
    if np.random.random() < 0.3:
        seq = _scale(seq)
    return seq


# ── Dataset ───────────────────────────────────────────────────────────────────

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, classes, augment_data=False, use_relative=False):
        """
        Args:
            data_dir:     path to data/ folder
            classes:      list of class names (folder names)
            augment_data: apply augmentation (True for train, False for val)
            use_relative: convert to relative coords via to_relative_holistic()
        """
        self.samples      = []
        self.augment_data = augment_data
        self.use_relative = use_relative
        self.label_map    = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in sorted(os.listdir(cls_dir)):
                if file.endswith(".npz"):
                    self.samples.append(
                        (os.path.join(cls_dir, file), self.label_map[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path)["frames"].astype(np.float32)   # (30, 1662)

        # Convert to relative coords (uses hip center + shoulder scale as origin)
        # This makes the model invariant to signer position and distance
        if self.use_relative:
            frames = to_relative_holistic(frames)

        # Apply augmentation only during training
        if self.augment_data:
            frames = augment(frames)

        x = torch.tensor(frames, dtype=torch.float32)
        y = torch.tensor(label,  dtype=torch.long)
        return x, y