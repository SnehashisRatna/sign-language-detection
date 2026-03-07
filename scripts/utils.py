# scripts/utils.py

import os
import numpy as np


# ─────────────────────────────────────────────
# LANDMARK CONFIGURATION
# ─────────────────────────────────────────────

FACE_N  = 468    # MediaPipe face landmarks
POSE_N  = 33     # MediaPipe pose landmarks
HAND_N  = 21     # MediaPipe hand landmarks (per hand)

# Feature sizes
FACE_FEAT  = FACE_N * 3        # 468 × 3  = 1404  (x, y, z)
POSE_FEAT  = POSE_N * 4        # 33  × 4  =  132  (x, y, z, visibility)
HANDS_FEAT = HAND_N * 3 * 2    # 21  × 3  × 2 =  126  (left + right)

# Total feature vector size = 1404 + 132 + 126 = 1662
FEAT_SIZE_HOLISTIC = FACE_FEAT + POSE_FEAT + HANDS_FEAT

# Slice indices within 1662-dim vector
FACE_START  = 0
FACE_END    = FACE_FEAT                          # 0    → 1404
POSE_START  = FACE_END                           # 1404
POSE_END    = FACE_END + POSE_FEAT               # 1404 → 1536
LH_START    = POSE_END                           # 1536
LH_END      = POSE_END + HAND_N * 3             # 1536 → 1599
RH_START    = LH_END                             # 1599
RH_END      = FEAT_SIZE_HOLISTIC                 # 1599 → 1662

# Default flags
USE_FACE_DEFAULT  = True
USE_POSE_DEFAULT  = True
USE_HANDS_DEFAULT = True


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_holistic_landmarks(
    results,
    use_face=USE_FACE_DEFAULT,
    use_pose=USE_POSE_DEFAULT,
    use_hands=USE_HANDS_DEFAULT
):
    """
    Convert MediaPipe Holistic results into a flat feature vector.

    Layout:
        Face  : 468 × 3 = 1404  (x, y, z)
        Pose  :  33 × 4 =  132  (x, y, z, visibility)
        L.Hand:  21 × 3 =   63  (x, y, z)
        R.Hand:  21 × 3 =   63  (x, y, z)
        Total :            1662

    Returns:
        np.ndarray of shape (1662,)
    """
    parts = []

    # ── Face ─────────────────────────────────
    if use_face:
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z])
        else:
            parts.extend([0.0] * FACE_FEAT)

    # ── Pose ─────────────────────────────────
    if use_pose:
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                parts.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            parts.extend([0.0] * POSE_FEAT)

    # ── Hands ─────────────────────────────────
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


# ─────────────────────────────────────────────
# RELATIVE COORDINATE CONVERSION  (FIXED)
# ─────────────────────────────────────────────

def to_relative_holistic(frames: np.ndarray) -> np.ndarray:
    """
    Convert absolute MediaPipe coordinates to body-relative coordinates.

    Origin : midpoint between left and right hip
    Scale  : shoulder width (distance between left and right shoulder)

    This makes landmarks invariant to:
      - Signer position in frame
      - Distance from camera
      - Body size differences between signers

    Args:
        frames: np.ndarray of shape (T, 1662)

    Returns:
        np.ndarray of shape (T, 1662) — normalized
    
    Fix over original:
        Original code did row.reshape(-1, 3) on the FULL 1662-dim vector,
        which is WRONG because pose landmarks have 4 values (x,y,z,visibility)
        not 3. This caused complete data corruption during training.
        
        Fixed by processing face, pose, and hands SEPARATELY with their
        correct value sizes.
    """
    frames = frames.copy().astype(np.float32)
    T      = len(frames)

    for t in range(T):
        row = frames[t]

        # ── Step 1: Extract pose block (4 values per landmark) ──
        pose = row[POSE_START:POSE_END].reshape(POSE_N, 4)

        # ── Step 2: Compute origin (hip center) ──────────────────
        try:
            left_hip  = pose[23, :3].copy()   # left hip  x,y,z
            right_hip = pose[24, :3].copy()   # right hip x,y,z
            origin    = (left_hip + right_hip) / 2.0
        except Exception:
            origin = np.zeros(3, dtype=np.float32)

        # ── Step 3: Compute scale (shoulder width) ────────────────
        try:
            left_sh  = pose[11, :3].copy()   # left shoulder
            right_sh = pose[12, :3].copy()   # right shoulder
            scale    = float(np.linalg.norm(left_sh - right_sh))
            if scale < 1e-6:
                scale = 1.0
        except Exception:
            scale = 1.0

        # ── Step 4: Normalize FACE (3 values per landmark) ────────
        face         = row[FACE_START:FACE_END].reshape(FACE_N, 3)
        face         = (face - origin) / (scale + 1e-8)
        frames[t, FACE_START:FACE_END] = face.flatten()

        # ── Step 5: Normalize POSE (only x,y,z — keep visibility) ─
        pose_xyz     = pose[:, :3].copy()
        pose_xyz     = (pose_xyz - origin) / (scale + 1e-8)
        pose[:, :3]  = pose_xyz
        frames[t, POSE_START:POSE_END] = pose.flatten()

        # ── Step 6: Normalize LEFT HAND (3 values per landmark) ───
        lh           = row[LH_START:LH_END].reshape(HAND_N, 3)
        lh           = (lh - origin) / (scale + 1e-8)
        frames[t, LH_START:LH_END] = lh.flatten()

        # ── Step 7: Normalize RIGHT HAND (3 values per landmark) ──
        rh           = row[RH_START:RH_END].reshape(HAND_N, 3)
        rh           = (rh - origin) / (scale + 1e-8)
        frames[t, RH_START:RH_END] = rh.flatten()

    return frames


# ─────────────────────────────────────────────
# DATASET NORMALIZATION
# ─────────────────────────────────────────────

def normalize_dataset(X: np.ndarray):
    """
    Zero-mean, unit-variance normalization across full dataset.

    Args:
        X: np.ndarray of shape (N, T, F)

    Returns:
        Xn   : normalized array
        mean : per-feature mean
        std  : per-feature std
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
    Xn   = (X - mean) / std
    return Xn, mean, std


# ─────────────────────────────────────────────
# SAVE SEQUENCE
# ─────────────────────────────────────────────

def save_sequence(save_dir: str, label: str, seq_idx, frames: np.ndarray) -> str:
    """
    Save a (30, 1662) sequence as a compressed .npz file.

    Saved to:
        data/<LABEL>/<LABEL>_<seq_idx>.npz

    Returns:
        Full path of saved file
    """
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    path = os.path.join(label_dir, f"{label}_{seq_idx}.npz")
    np.savez_compressed(path, frames=frames, label=label)
    return path


# ─────────────────────────────────────────────
# AUTO CLASS LOADER
# ─────────────────────────────────────────────

def get_classes_from_data(data_dir: str) -> list:
    """
    Automatically detect class folders inside data directory.
    Returns sorted list of class names.
    """
    classes = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    classes.sort()
    return classes


# ─────────────────────────────────────────────
# QUICK VERIFY — run: python scripts/utils.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'─'*45}")
    print(f"  Feature Vector Breakdown")
    print(f"{'─'*45}")
    print(f"  Face  : {FACE_N} landmarks × 3 = {FACE_FEAT}")
    print(f"  Pose  : {POSE_N} landmarks × 4 = {POSE_FEAT}")
    print(f"  Hands : {HAND_N} × 3 × 2      = {HANDS_FEAT}")
    print(f"  {'─'*30}")
    print(f"  TOTAL : {FEAT_SIZE_HOLISTIC}")
    print(f"{'─'*45}\n")

    # Test to_relative_holistic with dummy data
    dummy = np.random.rand(30, FEAT_SIZE_HOLISTIC).astype(np.float32)
    out   = to_relative_holistic(dummy)
    assert out.shape == (30, FEAT_SIZE_HOLISTIC), "Shape mismatch!"
    print("  ✅ to_relative_holistic() — shape check passed")
    print(f"  Input  mean: {dummy.mean():.4f}")
    print(f"  Output mean: {out.mean():.4f}  (should be near 0)")
    print(f"\n{'─'*45}\n")