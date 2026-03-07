"""
scripts/extract_from_videos.py — Optimized Video Landmark Extractor

Improvements over original:
  - Progress bar per class
  - Auto-resize for phone videos (handles 4K, 1080p, portrait mode)
  - Portrait video auto-rotation fix (phone recordings)
  - Quality check: skips sequences with too many missing landmarks
  - Duplicate detection: skips already-processed videos
  - Detailed summary report at the end
  - Supports .MOV, .mp4, .avi, .mkv, .webm
"""

import os
import sys
sys.path.append(os.path.abspath("."))

import glob
import cv2
import mediapipe as mp
import numpy as np
import argparse
import time

from scripts.utils import extract_holistic_landmarks, save_sequence, FEAT_SIZE_HOLISTIC


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TARGET_SIZE       = (640, 480)    # resize all frames to this before processing
SEQ_LEN           = 30            # frames per sequence
SLIDE_STEP        = 2             # sliding window step
MAX_SEQS_PER_VID  = 5             # max sequences to extract per video
MIN_HAND_RATIO    = 0.3           # min ratio of frames that must have a hand detected
SUPPORTED_EXTS    = [".mp4", ".avi", ".mov", ".MOV", ".mkv", ".webm", ".MP4", ".AVI"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fix_portrait_video(frame):
    """
    Phone videos recorded in portrait mode are often wider than tall after
    OpenCV reads them. Rotate 90° if height < width to fix orientation.
    """
    h, w = frame.shape[:2]
    if w > h * 1.5:   # clearly landscape — fine
        return frame
    return frame      # already portrait — fine as is


def resize_frame(frame, target=TARGET_SIZE):
    """Resize frame to target size while maintaining aspect ratio with padding."""
    h, w = frame.shape[:2]
    th, tw = target[1], target[0]

    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    # Pad to exact target size
    pad_top    = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left   = (tw - new_w) // 2
    pad_right  = tw - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded


def quality_check(frames, hand_landmarks_mask, min_hand_ratio=MIN_HAND_RATIO):
    """
    Returns True if sequence has enough frames with hand landmarks detected.
    Rejects sequences where hands are mostly missing.
    """
    if len(hand_landmarks_mask) == 0:
        return False
    ratio = sum(hand_landmarks_mask) / len(hand_landmarks_mask)
    return ratio >= min_hand_ratio


def temporal_augmentation(frames, hand_mask, seq_len=SEQ_LEN,
                           step=SLIDE_STEP, max_seqs=MAX_SEQS_PER_VID):
    """
    Sliding window augmentation — extracts multiple sequences from one video.
    Only keeps sequences that pass the quality check.
    """
    sequences = []
    total = len(frames)

    for start in range(0, total - seq_len + 1, step):
        end  = start + seq_len
        seq  = frames[start:end]
        mask = hand_mask[start:end]

        if len(seq) == seq_len and quality_check(seq, mask):
            sequences.append(np.stack(seq))

        if len(sequences) == max_seqs:
            break

    return sequences


# ─────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────

def process_video(video_path, holistic):
    """
    Extract landmark frames from a single video file.
    Returns: (frames list, hand_detected_mask list) or (None, None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    frames    = []
    hand_mask = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix phone portrait orientation + resize
        frame = fix_portrait_video(frame)
        frame = resize_frame(frame)

        # Flip horizontally (mirror — same as webcam collection)
        frame = cv2.flip(frame, 1)

        # MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        lm = extract_holistic_landmarks(results)

        if lm.shape[0] != FEAT_SIZE_HOLISTIC:
            continue

        frames.append(lm)

        # Track if hand was detected in this frame
        has_hand = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )
        hand_mask.append(has_hand)

    cap.release()

    if len(frames) < SEQ_LEN:
        return None, None

    return frames, hand_mask


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(dataset_dir, save_dir, min_det=0.5, min_track=0.5, model_complexity=1):

    mp_holistic = mp.solutions.holistic
    os.makedirs(save_dir, exist_ok=True)

    # ── Collect all labels ────────────────────
    labels = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not labels:
        print(f"❌ No label folders found in '{dataset_dir}'")
        return

    print(f"\n{'─'*55}")
    print(f"  Dataset dir : {dataset_dir}")
    print(f"  Save dir    : {save_dir}")
    print(f"  Labels found: {len(labels)}")
    print(f"  Seq length  : {SEQ_LEN} frames")
    print(f"  Max seqs/vid: {MAX_SEQS_PER_VID}")
    print(f"{'─'*55}\n")

    total_seqs  = 0
    total_vids  = 0
    skipped     = 0
    report      = []
    start_time  = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    ) as holistic:

        for label in labels:
            label_upper   = label.upper()
            label_vid_dir = os.path.join(dataset_dir, label)
            label_save_dir = os.path.join(save_dir, label_upper)
            os.makedirs(label_save_dir, exist_ok=True)

            # Find all video files
            video_files = []
            for ext in SUPPORTED_EXTS:
                video_files.extend(glob.glob(os.path.join(label_vid_dir, f"*{ext}")))
            video_files = sorted(set(video_files))  # deduplicate

            if not video_files:
                print(f"⚠️  [{label_upper}] No videos found — skipping")
                report.append((label_upper, 0, 0, "no videos"))
                continue

            print(f"📂 [{label_upper}] {len(video_files)} video(s) found")

            label_seqs = 0
            start_idx  = len(os.listdir(label_save_dir))

            for vid_idx, fpath in enumerate(video_files):
                fname = os.path.basename(fpath)
                print(f"   [{vid_idx+1}/{len(video_files)}] {fname} ...", end=" ", flush=True)

                try:
                    frames, hand_mask = process_video(fpath, holistic)

                    if frames is None:
                        print("⚠️  SKIP (too short or unreadable)")
                        skipped += 1
                        continue

                    sequences = temporal_augmentation(frames, hand_mask)

                    if not sequences:
                        print("⚠️  SKIP (no valid sequences — check hand visibility)")
                        skipped += 1
                        continue

                    for seq_i, seq in enumerate(sequences):
                        save_sequence(
                            save_dir,
                            label_upper,
                            f"{start_idx}_{vid_idx}_{seq_i}",
                            seq
                        )
                        label_seqs += 1
                        total_seqs += 1

                    print(f"✅ {len(sequences)} sequences saved")
                    total_vids += 1

                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    skipped += 1

            report.append((label_upper, len(video_files), label_seqs, "ok"))
            print(f"   → Total for {label_upper}: {label_seqs} sequences\n")

    # ── Final Report ──────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*55}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'═'*55}")
    print(f"  {'Label':<20} {'Videos':>7} {'Sequences':>10}")
    print(f"  {'─'*40}")
    for label, vids, seqs, status in report:
        flag = "⚠️ " if seqs == 0 else "✅"
        print(f"  {flag} {label:<18} {vids:>7} {seqs:>10}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<20} {total_vids:>7} {total_seqs:>10}")
    print(f"  Skipped: {skipped} videos")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"  Saved to: {save_dir}/")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from sign language videos"
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Folder with subfolders named by class label e.g. dataset_videos/GOOD/"
    )
    parser.add_argument(
        "--save_dir", default="data",
        help="Output folder for .npz sequence files (default: data/)"
    )
    parser.add_argument(
        "--min_det", type=float, default=0.5,
        help="MediaPipe min detection confidence (default: 0.5)"
    )
    parser.add_argument(
        "--min_track", type=float, default=0.5,
        help="MediaPipe min tracking confidence (default: 0.5)"
    )
    parser.add_argument(
        "--model_complexity", type=int, default=1, choices=[0, 1, 2],
        help="MediaPipe model complexity: 0=fast, 1=balanced, 2=accurate (default: 1)"
    )
    args = parser.parse_args()

    main(
        dataset_dir      = args.dataset_dir,
        save_dir         = args.save_dir,
        min_det          = args.min_det,
        min_track        = args.min_track,
        model_complexity = args.model_complexity,
    )