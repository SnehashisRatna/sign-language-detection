"""
extract_from_videos.py (MediaPipe Holistic + Temporal Augmentation)

Convert videos organized by label into multiple .npz sequences
using temporal sliding-window augmentation.
"""

import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import argparse

from utils import extract_holistic_landmarks, save_sequence, FEAT_SIZE_HOLISTIC


# -------------------------
# TEMPORAL AUGMENTATION
# -------------------------
def temporal_augmentation(frames, seq_len=30, step=2, max_seqs=4):
    """
    frames: list of (1662,) arrays
    returns: list of (30, 1662) sequences
    """
    sequences = []
    total_frames = len(frames)

    for start in range(0, total_frames - seq_len + 1, step):
        end = start + seq_len
        seq = frames[start:end]

        if len(seq) == seq_len:
            sequences.append(np.stack(seq))

        if len(sequences) == max_seqs:
            break

    return sequences


def process_video_file(video_path, holistic):
    """Return list of landmark frames extracted from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        lm = extract_holistic_landmarks(res)

        if lm.shape[0] != FEAT_SIZE_HOLISTIC:
            raise ValueError(
                f"Feature size mismatch: {lm.shape[0]} != {FEAT_SIZE_HOLISTIC}"
            )

        frames.append(lm)

    cap.release()

    return frames if len(frames) > 0 else None


def main(dataset_dir, save_dir, seq_len, ext_list,
         min_det=0.5, min_track=0.5, model_complexity=1, refine_face=False):

    mp_holistic = mp.solutions.holistic
    os.makedirs(save_dir, exist_ok=True)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        refine_face_landmarks=refine_face,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track
    ) as holistic:

        total = 0

        for label in sorted(os.listdir(dataset_dir)):
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue

            label = label.upper()

            files = []
            for ext in ext_list:
                files.extend(glob.glob(os.path.join(label_dir, f"*{ext}")))
            files = sorted(files)

            if not files:
                print(f"[WARN] No videos for label '{label}'")
                continue

            print(f"[INFO] Processing '{label}' ({len(files)} videos)")

            label_save_dir = os.path.join(save_dir, label)
            os.makedirs(label_save_dir, exist_ok=True)
            start_idx = len(os.listdir(label_save_dir))

            for vid_idx, fpath in enumerate(files):
                print(f"  -> {os.path.basename(fpath)}")

                try:
                    frames = process_video_file(fpath, holistic)
                    if frames is None:
                        print("     [SKIP] Empty video")
                        continue

                    sequences = temporal_augmentation(
                        frames,
                        seq_len=seq_len,
                        step=2,
                        max_seqs=4
                    )

                    for seq_i, seq in enumerate(sequences):
                        save_path = save_sequence(
                            save_dir,
                            label,
                            f"{start_idx}_{vid_idx}_{seq_i}",
                            seq
                        )
                        total += 1
                        print(f"     Saved: {save_path}")

                except Exception as e:
                    print(f"     [ERROR] {e}")

        print(f"[DONE] Extracted {total} sequences into '{save_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--save_dir", default="data")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--ext", nargs="+",
                        default=[".mp4", ".avi", ".mov", ".mkv"])
    parser.add_argument("--min_det", type=float, default=0.5)
    parser.add_argument("--min_track", type=float, default=0.5)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--refine_face", action="store_true")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.save_dir,
        args.seq_len,
        args.ext,
        min_det=args.min_det,
        min_track=args.min_track,
        model_complexity=args.model_complexity,
        refine_face=args.refine_face
    )
