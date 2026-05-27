"""
scripts/extract_from_videos.py — Optimized Video Landmark Extractor

Key feature: Auto-clips videos between CLIP_START and CLIP_END seconds
so only the middle section (actual sign) is used for extraction.

Fixes applied:
  [1] start_idx overwrite bug fixed — now uses run_id (timestamp-based)
  [2] --no_flip flag added — for side/angle view videos
  [3] MIN_HAND_RATIO raised to 0.5 — cleaner sequences for weak classes
  [4] Re-run protection — skips already-processed videos
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
TARGET_SIZE      = (640, 480)   # resize all frames to this
SEQ_LEN          = 30           # frames per sequence
SLIDE_STEP       = 5            # sliding window step
MAX_SEQS_PER_VID = 10            # max sequences per video
MIN_HAND_RATIO   = 0.5          # FIX [3]: raised from 0.3 → 0.5 for cleaner sequences
CLIP_START_SEC   = 3.0          # skip first N seconds of video
CLIP_END_SEC     = 7.0          # stop at this second of video
SUPPORTED_EXTS   = [".mp4", ".avi", ".mov", ".MOV", ".mkv", ".webm", ".MP4", ".AVI"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def resize_frame(frame, target=TARGET_SIZE):
    """Resize frame maintaining aspect ratio with black padding."""
    h, w   = frame.shape[:2]
    th, tw = target[1], target[0]
    scale  = min(tw / w, th / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)

    resized    = cv2.resize(frame, (new_w, new_h))
    pad_top    = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left   = (tw - new_w) // 2
    pad_right  = tw - new_w - pad_left

    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def quality_check(hand_mask, min_ratio=MIN_HAND_RATIO):
    """Returns True if enough frames have hand landmarks detected."""
    if not hand_mask:
        return False
    return sum(hand_mask) / len(hand_mask) >= min_ratio


def temporal_augmentation(frames, hand_mask, seq_len=SEQ_LEN,
                           step=SLIDE_STEP, max_seqs=MAX_SEQS_PER_VID):
    """Sliding window — extract multiple sequences from clipped frames."""
    sequences = []
    total     = len(frames)

    for start in range(0, total - seq_len + 1, step):
        end  = start + seq_len
        seq  = frames[start:end]
        mask = hand_mask[start:end]

        if len(seq) == seq_len and quality_check(mask):
            sequences.append(np.stack(seq))

        if len(sequences) == max_seqs:
            break

    return sequences


# ─────────────────────────────────────────────
# CORE: Process single video with auto-clip
# ─────────────────────────────────────────────

def process_video(video_path, holistic,
                  clip_start=CLIP_START_SEC,
                  clip_end=CLIP_END_SEC,
                  flip=True):                   # FIX [2]: flip flag added
    """
    Extract landmark frames from CLIP_START to CLIP_END seconds only.
    Skips the entry/exit portions of the video automatically.

    Args:
        flip: True  → mirror horizontally (front-view / webcam-style videos)
              False → no flip (side-angle / diagonal videos)

    Returns: (frames, hand_mask) or (None, None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        return None, None

    # Calculate frame range for clip_start → clip_end
    start_frame = int(clip_start * fps)
    end_frame   = int(clip_end   * fps)

    # Safety: clamp to actual video length
    start_frame = min(start_frame, total_frames - 1)
    end_frame   = min(end_frame,   total_frames)

    if end_frame <= start_frame:
        cap.release()
        return None, None

    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames    = []
    hand_mask = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)

        # FIX [2]: Only flip for front-view videos (matches webcam collection)
        #          Skip flip for angle/side videos to preserve correct geometry
        if flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        lm = extract_holistic_landmarks(results)

        if lm.shape[0] != FEAT_SIZE_HOLISTIC:
            frame_idx += 1
            continue

        frames.append(lm)

        has_hand = (
            results.left_hand_landmarks  is not None or
            results.right_hand_landmarks is not None
        )
        hand_mask.append(has_hand)
        frame_idx += 1

    cap.release()

    if len(frames) < SEQ_LEN:
        return None, None

    return frames, hand_mask


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(dataset_dir, save_dir, clip_start, clip_end,
         flip=True, min_det=0.5, min_track=0.5, model_complexity=1):

    mp_holistic = mp.solutions.holistic
    os.makedirs(save_dir, exist_ok=True)

    labels = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not labels:
        print(f"❌ No label folders found in '{dataset_dir}'")
        return

    # FIX [1]: Unique run ID per execution — prevents filename collisions on re-runs
    run_id = int(time.time())

    print(f"\n{'─'*55}")
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  Save dir     : {save_dir}")
    print(f"  Labels found : {len(labels)}")
    print(f"  Clip window  : {clip_start}s → {clip_end}s")
    print(f"  Seq length   : {SEQ_LEN} frames")
    print(f"  Max seqs/vid : {MAX_SEQS_PER_VID}")
    print(f"  Flip frames  : {'YES (front-view mode)' if flip else 'NO (angle-view mode)'}")
    print(f"  Run ID       : {run_id}")
    print(f"{'─'*55}\n")

    total_seqs = 0
    total_vids = 0
    skipped    = 0
    report     = []
    start_time = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    ) as holistic:

        for label in labels:
            label_upper    = label.upper()
            label_vid_dir  = os.path.join(dataset_dir, label)
            label_save_dir = os.path.join(save_dir, label_upper)
            os.makedirs(label_save_dir, exist_ok=True)

            # Find all video files
            video_files = []
            for ext in SUPPORTED_EXTS:
                video_files.extend(
                    glob.glob(os.path.join(label_vid_dir, f"*{ext}"))
                )
            video_files = sorted(set(video_files))

            if not video_files:
                print(f"⚠️  [{label_upper}] No videos found — skipping")
                report.append((label_upper, 0, 0))
                continue

            print(f"📂 [{label_upper}] {len(video_files)} video(s)")

            label_seqs = 0

            # FIX [4]: Build set of already-processed video basenames
            existing_files  = set(os.listdir(label_save_dir))
            processed_names = set()
            for f in existing_files:
                # filenames are like: {run_id}_{vid_idx}_{seq_i}.npz
                # track by stem to detect previously processed videos
                parts = f.replace(".npz", "").split("_")
                if len(parts) >= 2:
                    processed_names.add(parts[1])  # vid_idx portion

            for vid_idx, fpath in enumerate(video_files):
                fname      = os.path.basename(fpath)
                fname_stem = os.path.splitext(fname)[0]

                print(f"   [{vid_idx+1}/{len(video_files)}] {fname} ...",
                      end=" ", flush=True)

                # FIX [4]: Skip if this video was already processed in a prior run
                if fname_stem in existing_files or \
                   any(fname_stem in f for f in existing_files):
                    print("⏩ SKIP — already processed")
                    continue

                try:
                    frames, hand_mask = process_video(
                        fpath, holistic,
                        clip_start=clip_start,
                        clip_end=clip_end,
                        flip=flip              # FIX [2]: pass flip flag through
                    )

                    if frames is None:
                        print("⚠️  SKIP — too short or hands not visible")
                        skipped += 1
                        continue

                    sequences = temporal_augmentation(frames, hand_mask)

                    if not sequences:
                        print("⚠️  SKIP — no valid sequences extracted")
                        skipped += 1
                        continue

                    # FIX [1]: Use run_id + fname_stem instead of start_idx
                    #           Guarantees unique filenames across re-runs
                    for seq_i, seq in enumerate(sequences):
                        save_sequence(
                            save_dir,
                            label_upper,
                            f"{run_id}_{fname_stem}_{seq_i}",
                            seq
                        )
                        label_seqs += 1
                        total_seqs += 1

                    print(f"✅ {len(sequences)} sequences")
                    total_vids += 1

                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    skipped += 1

            report.append((label_upper, len(video_files), label_seqs))
            print(f"   → {label_upper}: {label_seqs} sequences\n")

    # ── Summary Report ────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*50}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'═'*50}")
    print(f"  {'Label':<20} {'Videos':>7} {'Seqs':>8}")
    print(f"  {'─'*38}")
    for label, vids, seqs in report:
        flag = "⚠️ " if seqs == 0 else "✅"
        print(f"  {flag} {label:<18} {vids:>7} {seqs:>8}")
    print(f"  {'─'*38}")
    print(f"  {'TOTAL':<20} {total_vids:>7} {total_seqs:>8}")
    print(f"\n  Skipped : {skipped} videos")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"  Saved to: {save_dir}/")
    print(f"{'═'*50}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from sign language videos"
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Root folder containing subfolders named by class e.g. dataset_videos/GOOD/"
    )
    parser.add_argument(
        "--save_dir", default="data",
        help="Output folder for .npz files (default: data/)"
    )
    parser.add_argument(
        "--clip_start", type=float, default=3.0,
        help="Start extracting from this second (default: 3.0)"
    )
    parser.add_argument(
        "--clip_end", type=float, default=7.0,
        help="Stop extracting at this second (default: 7.0)"
    )
    # FIX [2]: --no_flip flag for side/angle/diagonal view videos
    parser.add_argument(
        "--no_flip", action="store_true",
        help="Disable horizontal flip — use this for side/angle view videos"
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
        help="MediaPipe complexity: 0=fast, 1=balanced, 2=accurate (default: 1)"
    )
    args = parser.parse_args()

    main(
        dataset_dir      = args.dataset_dir,
        save_dir         = args.save_dir,
        clip_start       = args.clip_start,
        clip_end         = args.clip_end,
        flip             = not args.no_flip,   # FIX [2]: invert the flag
        min_det          = args.min_det,
        min_track        = args.min_track,
        model_complexity = args.model_complexity,
    )