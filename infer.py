# infer.py

import os
import sys
sys.path.append(os.path.abspath("."))   # 🔥 Fix for Windows import issue

print("🚀 infer.py started")

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter

from models.gru_model import GRUClassifier
from scripts.utils import (
    extract_holistic_landmarks,
    FEAT_SIZE_HOLISTIC,
    get_classes_from_data
)

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "gru_sanity_model.pth"
DATA_DIR = "data"

SEQ_LEN = 30
CONF_THRESH = 0.60
DEVICE = "cpu"

PRED_EVERY_N_FRAMES = 3
VOTE_WINDOW = 6
UNKNOWN_LABEL = "Unknown"

# -------------------------
# LOAD CLASSES
# -------------------------
CLASSES = get_classes_from_data(DATA_DIR)

print("✅ Classes loaded:", len(CLASSES))

# -------------------------
# LOAD MODEL
# -------------------------
model = GRUClassifier(
    input_size=1662,
    hidden_size=128,
    num_classes=len(CLASSES)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")


# -------------------------
# REAL-TIME INFERENCE
# -------------------------
def realtime_inference():

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    frame_buffer = deque(maxlen=SEQ_LEN)
    pred_history = deque(maxlen=VOTE_WINDOW)
    frame_count = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = extract_holistic_landmarks(results)

            if landmarks.shape[0] == FEAT_SIZE_HOLISTIC:
                frame_buffer.append(landmarks)

            prediction = UNKNOWN_LABEL
            confidence = 0.0

            if len(frame_buffer) == SEQ_LEN and frame_count % PRED_EVERY_N_FRAMES == 0:

                x = torch.tensor(
                    np.array(frame_buffer),
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)
                    conf, pred = torch.max(probs, dim=1)

                    confidence = conf.item()
                    label = CLASSES[pred.item()]

                    if confidence >= CONF_THRESH:
                        pred_history.append(label)
                    else:
                        pred_history.append(UNKNOWN_LABEL)

            if len(pred_history) > 0:
                most_common = Counter(pred_history).most_common(1)[0][0]
                if most_common != UNKNOWN_LABEL:
                    prediction = most_common

            cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)

            cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Sign Language Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_inference()