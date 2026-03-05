# utils.py

import os
import numpy as np


# -------------------------
# LANDMARK CONFIGURATION
# -------------------------

FACE_N = 468
POSE_N = 33
HAND_N = 21

USE_FACE_DEFAULT = True
USE_POSE_DEFAULT = True
USE_HANDS_DEFAULT = True

FACE_FEAT = FACE_N * 3
POSE_FEAT = POSE_N * 4
HANDS_FEAT = HAND_N * 3 * 2

FEAT_SIZE_HOLISTIC = (
    (FACE_FEAT if USE_FACE_DEFAULT else 0) +
    (POSE_FEAT if USE_POSE_DEFAULT else 0) +
    (HANDS_FEAT if USE_HANDS_DEFAULT else 0)
)


# -------------------------
# FEATURE EXTRACTION
# -------------------------

def extract_holistic_landmarks(
    results,
    use_face=USE_FACE_DEFAULT,
    use_pose=USE_POSE_DEFAULT,
    use_hands=USE_HANDS_DEFAULT
):
    """
    Convert MediaPipe Holistic results into a flattened feature vector.
    Output shape: (1662,)
    """

    parts = []

    # FACE
    if use_face:
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z])
        else:
            parts.extend([0.0] * FACE_FEAT)

    # POSE
    if use_pose:
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            parts.extend([0.0] * POSE_FEAT)

    # HANDS
    if use_hands:

        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z])
        else:
            parts.extend([0.0] * (HAND_N * 3))

        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z])
        else:
            parts.extend([0.0] * (HAND_N * 3))

    return np.array(parts, dtype=np.float32)


# -------------------------
# RELATIVE COORDINATE CONVERSION
# -------------------------

def to_relative_holistic(frames):
    """
    Convert absolute coordinates to relative coordinates
    using hip center as origin and shoulder width as scale.
    """

    frames = frames.copy()
    T, F = frames.shape

    pose_offset = FACE_FEAT if USE_FACE_DEFAULT else 0

    for t in range(T):
        row = frames[t]

        pose = row[pose_offset:pose_offset + POSE_FEAT].reshape(-1, 4)

        try:
            left_hip = pose[23][:3]
            right_hip = pose[24][:3]
            origin = (left_hip + right_hip) / 2.0
        except:
            origin = np.zeros(3)

        try:
            left_sh = pose[11][:3]
            right_sh = pose[12][:3]
            scale = np.linalg.norm(left_sh - right_sh)
            if scale < 1e-6:
                scale = 1.0
        except:
            scale = 1.0

        coords = row.reshape(-1, 3)
        coords = (coords - origin) / (scale + 1e-8)
        frames[t] = coords.flatten()

    return frames


# -------------------------
# DATASET NORMALIZATION
# -------------------------

def normalize_dataset(X):
    """
    Zero-mean, unit-variance normalization.
    X shape: (N, T, F)
    """

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    Xn = (X - mean) / std
    return Xn, mean, std


# -------------------------
# SAVE SEQUENCE
# -------------------------

def save_sequence(save_dir, label, seq_idx, frames):
    """
    Save sequence as:
      data/<LABEL>/<LABEL>_<idx>.npz
    """

    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    path = os.path.join(label_dir, f"{label}_{seq_idx}.npz")
    np.savez_compressed(path, frames=frames, label=label)
    return path


# -------------------------
# AUTO CLASS LOADER
# -------------------------

def get_classes_from_data(data_dir):
    """
    Automatically detect class folders inside data directory.
    """

    classes = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    classes.sort()
    return classes