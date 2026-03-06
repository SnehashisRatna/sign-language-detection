# infer.py

import os
import sys
sys.path.append(os.path.abspath("."))

print("🚀 infer.py started")

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import ImageFont, ImageDraw, Image

from models.gru_model import GRUClassifier
from scripts.utils import (
    extract_holistic_landmarks,
    FEAT_SIZE_HOLISTIC,
    get_classes_from_data,
)
from scripts.odia_labels import get_odia


# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH      = "gru_best_model.pth"
DATA_DIR        = "data"
FONT_PATH       = "assets/NotoSansOriya-Regular.ttf"

SEQ_LEN         = 30
DEVICE          = "cpu"

PRED_EVERY_N    = 2
SMOOTH_WINDOW   = 6
CONF_THRESH     = 0.50
DEBOUNCE_FRAMES = 6

ENGLISH_FONT_SIZE = 52
ODIA_FONT_SIZE    = 52


# ─────────────────────────────────────────────
# LOAD FONT
# ─────────────────────────────────────────────
try:
    odia_font = ImageFont.truetype(FONT_PATH, ODIA_FONT_SIZE)
    print(f"✅ Odia font loaded")
except Exception as e:
    print(f"❌ Font error: {e}")
    odia_font = None


# ─────────────────────────────────────────────
# LOAD CLASSES + MODEL
# ─────────────────────────────────────────────
CLASSES = get_classes_from_data(DATA_DIR)
print(f"✅ Classes loaded: {len(CLASSES)}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    CLASSES    = ckpt.get("classes", CLASSES)
    state_dict = ckpt["model_state"]
else:
    state_dict = ckpt

model = GRUClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.0,
).to(DEVICE)

model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded successfully\n")


# ─────────────────────────────────────────────
# ODIA TEXT RENDERER
# ─────────────────────────────────────────────
def put_odia_text(frame, text, position, font_size=ODIA_FONT_SIZE, color=(255, 255, 255)):
    if odia_font is None or not text:
        return frame
    img_pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(img_pil)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=odia_font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# DRAW LANDMARKS
# ─────────────────────────────────────────────
def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame, results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style()
    )


# ─────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(frame_buffer, prob_buffer):
    seq   = np.array(frame_buffer, dtype=np.float32)
    x     = torch.tensor(seq).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    prob_buffer.append(probs)
    avg_probs  = np.mean(prob_buffer, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])

    if confidence >= CONF_THRESH:
        return CLASSES[pred_idx], confidence
    return None, confidence


# ─────────────────────────────────────────────
# HUD DRAWING
# Layout:
#   TOP    — English word (large, white)
#   BOTTOM — Odia word (large, cyan)
# ─────────────────────────────────────────────
def draw_hud(frame, prediction, confidence, buf_len):
    h, w = frame.shape[:2]

    # ── TOP: English word ─────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    if prediction:
        english_text = f"{prediction}  ({confidence*100:.1f}%)"
        cv2.putText(frame, english_text, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 80), 3)
    else:
        cv2.putText(frame, "...", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2)

    # ── BOTTOM: Odia word ─────────────────────
    cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)

    if prediction:
        odia_text = get_odia(prediction)
        frame = put_odia_text(
            frame, odia_text,
            (10, h - 90),
            color=(0, 255, 255)   # cyan
        )

    # ── Confidence bar (green/red) ────────────
    conf_w    = int(confidence * (w - 20))
    bar_color = (0, 200, 0) if confidence >= CONF_THRESH else (0, 0, 200)
    cv2.rectangle(frame, (10, h - 20), (10 + conf_w, h - 12), bar_color, -1)
    cv2.rectangle(frame, (10, h - 20), (w - 10,      h - 12), (80, 80, 80), 1)

    # ── Buffer fill bar (orange) ──────────────
    fill_w = int((buf_len / SEQ_LEN) * (w - 20))
    cv2.rectangle(frame, (10, h - 8), (10 + fill_w, h - 2), (255, 160, 0), -1)
    cv2.rectangle(frame, (10, h - 8), (w - 10,      h - 2), (80, 80, 80), 1)

    return frame


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def realtime_inference():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_buffer = deque(maxlen=SEQ_LEN)
    prob_buffer  = deque(maxlen=SMOOTH_WINDOW)
    frame_count  = 0
    debounce     = 0
    last_pred    = None
    confidence   = 0.0

    print("📷 Camera running  —  press 'q' to quit\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)

            # ── MediaPipe ─────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Draw landmarks ────────────────
            draw_landmarks(frame, results)

            # ── Extract keypoints ─────────────
            landmarks = extract_holistic_landmarks(results)
            if landmarks.shape[0] == FEAT_SIZE_HOLISTIC:
                frame_buffer.append(landmarks)

            # ── Only predict when hand visible ─
            hands_detected = (
                results.left_hand_landmarks is not None or
                results.right_hand_landmarks is not None
            )

            if len(frame_buffer) == SEQ_LEN and frame_count % PRED_EVERY_N == 0 and hands_detected:
                last_pred, confidence = predict(frame_buffer, prob_buffer)

                debounce += 1
                if last_pred and debounce >= DEBOUNCE_FRAMES:
                    odia = get_odia(last_pred)
                    print(f"[Detected] {last_pred} → {odia}  ({confidence*100:.1f}%)")
                    debounce = 0

            elif not hands_detected:
                last_pred  = None
                confidence = 0.0
                prob_buffer.clear()

            # ── Draw HUD ──────────────────────
            frame = draw_hud(frame, last_pred, confidence, len(frame_buffer))

            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_inference()