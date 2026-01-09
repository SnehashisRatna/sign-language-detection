# infer.py
print("🚀 infer.py started")

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter

from models.gru_model import GRUClassifier
from scripts.utils import extract_holistic_landmarks, FEAT_SIZE_HOLISTIC

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "gru_sanity_model.pth"
CLASSES = ["HELLO", "SORRY", "THANKYOU"]
SEQ_LEN = 30
CONF_THRESH = 0.6
DEVICE = "cpu"

# -------------------------
# LOAD MODEL (MUST MATCH TRAINING)
# -------------------------
model = GRUClassifier(
    input_size=1662,
    hidden_size=128,     # ⚠️ MUST MATCH TRAINING
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
        print("❌ ERROR: Camera not accessible")
        return

    print("📷 Camera opened successfully")

    cv2.namedWindow("Sign Language Inference", cv2.WINDOW_NORMAL)

    frame_buffer = deque(maxlen=SEQ_LEN)
    pred_history = deque(maxlen=10)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = extract_holistic_landmarks(results)

            if landmarks.shape[0] == FEAT_SIZE_HOLISTIC:
                frame_buffer.append(landmarks)

            prediction = "Collecting..."
            confidence = 0.0

            if len(frame_buffer) == SEQ_LEN:
                seq = np.array(frame_buffer)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                    confidence = conf.item()
                    label = CLASSES[pred.item()]

                    if confidence >= CONF_THRESH:
                        pred_history.append(label)

                    if len(pred_history) > 0:
                        prediction = Counter(pred_history).most_common(1)[0][0]
                    else:
                        prediction = "Uncertain"

            # -------------------------
            # DISPLAY
            # -------------------------
            cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Sign Language Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("Choose mode:")
    print("2 → Real-time webcam inference")

    choice = input("Enter choice (2): ")

    if choice == "2":
        realtime_inference()
    else:
        print("Invalid choice")
