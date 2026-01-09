# scripts/utils.py
import os
import numpy as np


# Landmark counts (MediaPipe)

FACE_N = 468
POSE_N = 33
HAND_N = 21

# Toggle modalities
USE_FACE_DEFAULT = True
USE_POSE_DEFAULT = True
USE_HANDS_DEFAULT = True

# Feature sizes
FACE_FEAT = FACE_N * 3                  # 468 * (x,y,z)
POSE_FEAT = POSE_N * 4                  # 33 * (x,y,z,visibility)
HANDS_FEAT = HAND_N * 3 * 2             # left + right hand

# Total = 1404 + 132 + 126 = 1662
FEAT_SIZE_HOLISTIC = (
    (FACE_FEAT if USE_FACE_DEFAULT else 0) +
    (POSE_FEAT if USE_POSE_DEFAULT else 0) +
    (HANDS_FEAT if USE_HANDS_DEFAULT else 0)
)



# Feature Extraction

def extract_holistic_landmarks(results,
                               use_face=USE_FACE_DEFAULT,
                               use_pose=USE_POSE_DEFAULT,
                               use_hands=USE_HANDS_DEFAULT):
    """
    Convert MediaPipe Holistic results into a flattened feature vector.

    Ordering:
      [ face (468*3), pose (33*4), left hand (21*3), right hand (21*3) ]

    Missing landmarks are zero-padded.
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

    #  HANDS 
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



# Relative Coordinate Conversion

def to_relative_holistic(frames,
                         use_face=USE_FACE_DEFAULT,
                         use_pose=USE_POSE_DEFAULT,
                         use_hands=USE_HANDS_DEFAULT):
    """
    Convert absolute coordinates to relative coordinates.
    Uses hip center as origin and shoulder width as scale (if pose available).
    """
    frames = frames.copy()
    T, F = frames.shape

    offset = 0
    face_offset = offset if use_face else None
    if use_face:
        offset += FACE_FEAT

    pose_offset = offset if use_pose else None
    if use_pose:
        offset += POSE_FEAT

    hands_offset = offset if use_hands else None

    for t in range(T):
        row = frames[t].copy()
        origin = None
        scale = 1.0

        if use_pose and pose_offset is not None:
            pose = row[pose_offset:pose_offset + POSE_FEAT].reshape(-1, 4)
            try:
                left_hip = pose[23][:3]
                right_hip = pose[24][:3]
                origin = (left_hip + right_hip) / 2.0
            except Exception:
                origin = pose[0][:3]

            try:
                left_sh = pose[11][:3]
                right_sh = pose[12][:3]
                scale = np.linalg.norm(left_sh - right_sh)
                if scale < 1e-6:
                    scale = 1.0
            except Exception:
                scale = 1.0
        else:
            origin = np.zeros(3)
            scale = 1.0

        coords = row.reshape(-1, 3)
        coords = (coords - origin) / (scale + 1e-8)
        frames[t] = coords.flatten()

    return frames



# Dataset Normalization

def normalize_dataset(X):
    """
    Zero-mean, unit-variance normalization.
    X shape: (N, T, F)
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    Xn = (X - mean) / std
    return Xn, mean, std



# Save Sequence

def save_sequence(save_dir, label, seq_idx, frames):
    """
    Save sequence to:
      data/<LABEL>/<LABEL>_<idx>.npz

    frames shape: (T, 1662)
    """
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    path = os.path.join(label_dir, f"{label}_{seq_idx}.npz")
    np.savez_compressed(path, frames=frames, label=label)
    return path
