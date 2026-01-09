"""
collect_data.py — Updated with MediaPipe Holistic

Captures full body landmarks: face, pose, and both hands.
Each recording is saved as a compressed .npz file:
    <save_dir>/<label>_<seq_idx>.npz

Usage (from project root, with venv activated):
    python scripts/collect_data.py --label hello --seq_len 30 --seqs 60 --camera 0 --save_dir data

Controls while running:
    - Press 's' to start recording one sequence (records SEQ_LEN frames)
    - Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from utils import extract_holistic_landmarks, save_sequence


def collect(label, seq_len=30, seqs=60, camera_id=0, save_dir="data",
            min_detection_confidence=0.6, min_tracking_confidence=0.6):
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open camera {camera_id}. Try --camera 1 or 2.")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as holistic:

        collected = 0
        print(f"📹 Collecting for label '{label}' — Press 's' to record, 'q' to quit.")

        while collected < seqs:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            # Draw all detected landmarks
            if res.face_landmarks:
                mp_draw.draw_landmarks(
                    frame, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            if res.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.putText(frame, f"Label: {label}  Collected: {collected}/{seqs}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to record, 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Holistic Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                frames = []
                print(f"▶ Recording sequence {collected} for label '{label}' ...")

                for i in range(seq_len):
                    ret, f = cap.read()
                    if not ret:
                        break
                    f = cv2.flip(f, 1)
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    r = holistic.process(rgb)
                    lm = extract_holistic_landmarks(r)
                    frames.append(lm)

                    # Draw for visualization
                    if r.face_landmarks:
                        mp_draw.draw_landmarks(
                            f, r.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                    if r.pose_landmarks:
                        mp_draw.draw_landmarks(
                            f, r.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
                    if r.left_hand_landmarks:
                        mp_draw.draw_landmarks(f, r.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if r.right_hand_landmarks:
                        mp_draw.draw_landmarks(f, r.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    cv2.putText(f, f"Recording {i+1}/{seq_len}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Recording", f)
                    cv2.waitKey(30)

                if len(frames) == seq_len:
                    frames_np = np.stack(frames)
                    path = save_sequence(save_dir, label, collected, frames_np)
                    print(f"✅ Saved: {path}")
                    collected += 1
                else:
                    print("⚠ Recording interrupted or insufficient frames. Try again.")

            elif key == ord('q'):
                print("🛑 Quitting data collection.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help="Label name for the sequences (e.g., hello)")
    parser.add_argument('--seq_len', type=int, default=30, help="Frames per sequence")
    parser.add_argument('--seqs', type=int, default=60, help="Number of sequences to collect for this label")
    parser.add_argument('--camera', type=int, default=0, help="Camera device id (0,1,2...)")
    parser.add_argument('--save_dir', type=str, default='data', help="Directory to save .npz files")
    parser.add_argument('--min_detection_confidence', type=float, default=0.6)
    parser.add_argument('--min_tracking_confidence', type=float, default=0.6)
    args = parser.parse_args()

    collect(args.label, seq_len=args.seq_len, seqs=args.seqs, camera_id=args.camera,
            save_dir=args.save_dir, min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence)
