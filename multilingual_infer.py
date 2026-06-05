# multilingual_infer.py
#
# Real-Time Multilingual ISL Detection
#   Webcam → MediaPipe → extract_holistic_landmarks()
#          → frame_buffer (30 × 1662, raw)
#          → to_relative_holistic(seq)   ← called on full (30,1662) at predict time
#          → GRU model → English label
#          → Google Translate → Hindi / Odia / Tamil / Bengali / Telugu
#
# ── What is UNCHANGED from original infer.py ────────────────────────────────
#   • extract_holistic_landmarks()  — raw landmark extraction (utils.py)
#   • to_relative_holistic(seq)     — called on (30, 1662) array, not per-frame
#   • frame_buffer / prob_buffer    — deque-based, identical logic
#   • hands_detected gate           — only predict when hands are visible
#   • PRED_EVERY_N / SMOOTH_WINDOW  — same prediction cadence + smoothing
#   • checkpoint loading (key: "model_state", classes from ckpt)
#   • dropout=0.0 at inference
#   • cv2.flip(frame, 1)            — mirrored webcam view
#   • Pillow renderer for Odia script (put_odia_text / put_unicode_text)
#
# ── What is NEW ──────────────────────────────────────────────────────────────
#   • Google Translate (googletrans) — runtime multilingual output
#   • Keys 1-6 to switch language on the fly
#   • Language banner in HUD
#   • Per-language Noto font rendering for non-Latin scripts
#
# Key-bindings:
#   1 → English   2 → Hindi   3 → Odia
#   4 → Tamil     5 → Bengali  6 → Telugu
#   q → quit
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.abspath("."))

print("🚀 multilingual_infer.py started")

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import ImageFont, ImageDraw, Image

from models.gru_model import GRUClassifier
from scripts.utils import (
    extract_holistic_landmarks,   # ← same function as original infer.py
    to_relative_holistic,         # ← same function; called on (30,1662), not per-frame
    FEAT_SIZE_HOLISTIC,
    get_classes_from_data,
)
from scripts.odia_labels import get_odia
from googletrans import Translator


# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ─────────────────────────────────────────────
# CONFIG  (identical to original infer.py)
# ─────────────────────────────────────────────
MODEL_PATH      = "gru_best_model.pth"
DATA_DIR        = "data"

SEQ_LEN         = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_EVERY_N    = 2      # run model every N frames
SMOOTH_WINDOW   = 6      # average softmax probs over last N predictions
CONF_THRESH     = 0.50   # minimum confidence to display prediction
DEBOUNCE_FRAMES = 6      # min frames between terminal prints


# ─────────────────────────────────────────────
# LANGUAGE TABLE
# ─────────────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "Hindi":   "hi",
    "Odia":    "or",
    "Tamil":   "ta",
    "Bengali": "bn",
    "Telugu":  "te",
}

KEY_MAP = {
    ord('1'): "English",
    ord('2'): "Hindi",
    ord('3'): "Odia",
    ord('4'): "Tamil",
    ord('5'): "Bengali",
    ord('6'): "Telugu",
}

# Per-language Noto font paths
FONT_PATHS = {
    "en": None,
    "hi": "assets/fonts/NotoSansDevanagari-Regular.ttf",
    "or": "assets/fonts/NotoSansOriya-Regular.ttf",
    "ta": "assets/fonts/NotoSansTamil-Regular.ttf",
    "bn": "assets/fonts/NotoSansBengali-Regular.ttf",
    "te": "assets/fonts/NotoSansTelugu-Regular.ttf",
}
FONT_SIZE = 52   # px — same as original infer.py


# ─────────────────────────────────────────────
# RAQM CHECK
# Pillow needs the RAQM layout engine for Indic scripts (Odia, Hindi,
# Tamil, Bengali, Telugu) to render conjuncts and ligatures correctly.
# Without it, glyphs appear broken / unjoined.
# Fix: pip uninstall Pillow -y && pip install "Pillow[raqm]"
# ─────────────────────────────────────────────
try:
    _RAQM_LAYOUT = ImageFont.Layout.RAQM
    # Quick test — will raise if libraqm isn't actually installed
    _test_font_path = next(
        (p for p in FONT_PATHS.values() if p and os.path.exists(p)), None
    )
    if _test_font_path:
        ImageFont.truetype(_test_font_path, 12, layout_engine=_RAQM_LAYOUT)
    RAQM_AVAILABLE = True
    print("✅ RAQM layout engine available — Indic scripts will render correctly")
except Exception:
    RAQM_AVAILABLE = False
    print("⚠️  RAQM not available — Odia/Hindi/Tamil may render as broken glyphs")
    print("   Fix: pip uninstall Pillow -y && pip install \"Pillow[raqm]\"")


# ─────────────────────────────────────────────
# LOAD FONTS  (cached dict, one per language)
# Always tries RAQM first; falls back to basic truetype if unavailable.
# ─────────────────────────────────────────────
_font_cache: dict = {}

def _load_font(lang_code: str):
    """
    Return ImageFont for lang_code.
    Uses RAQM layout engine when available — required for correct
    rendering of Odia, Hindi, Tamil, Bengali, Telugu (Indic scripts).
    Falls back to basic layout if RAQM is not installed (glyphs will
    appear unjoined — install Pillow[raqm] to fix).
    """
    if lang_code in _font_cache:
        return _font_cache[lang_code]

    path = FONT_PATHS.get(lang_code, "")
    font = None

    if path and os.path.exists(path):
        try:
            if RAQM_AVAILABLE:
                font = ImageFont.truetype(
                    path, FONT_SIZE,
                    layout_engine=ImageFont.Layout.RAQM,   # ← correct shaping
                )
            else:
                font = ImageFont.truetype(path, FONT_SIZE)
            print(f"✅ Font loaded for '{lang_code}': {path}"
                  f"{'  [RAQM]' if RAQM_AVAILABLE else '  [basic — glyphs may look broken]'}")
        except Exception as exc:
            print(f"⚠️  Font load error for '{lang_code}': {exc}")
            font = None
    else:
        if lang_code != "en":
            print(f"⚠️  Font file not found for '{lang_code}': {path}")
            print(f"   Run: python download_noto_fonts.py")

    _font_cache[lang_code] = font
    return font

# Pre-load Odia font at startup
_load_font("or")


# ─────────────────────────────────────────────
# LOAD CLASSES + MODEL  (identical to original)
# ─────────────────────────────────────────────
CLASSES = get_classes_from_data(DATA_DIR)
print(f"✅ Classes loaded: {len(CLASSES)}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    CLASSES    = ckpt.get("classes", CLASSES)   # prefer classes saved in checkpoint
    state_dict = ckpt["model_state"]
else:
    state_dict = ckpt

model = GRUClassifier(
    input_size=1662,
    hidden_size=256,
    num_classes=len(CLASSES),
    dropout=0.0,    # ALWAYS 0.0 at inference — dropout only active during training
).to(DEVICE)

model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded successfully\n")


# ─────────────────────────────────────────────
# INIT TRANSLATOR
# ─────────────────────────────────────────────
translator = Translator()


# ─────────────────────────────────────────────
# UNICODE TEXT RENDERER  (Pillow — OpenCV can't render non-Latin scripts)
# Identical logic to put_odia_text() in original infer.py,
# generalised to accept any lang_code + font.
# ─────────────────────────────────────────────
def put_unicode_text(frame, text, position, lang_code, color=(0, 255, 255)):
    """
    Render Unicode text onto an OpenCV frame using Pillow.
    For English falls back to cv2.putText (no font file needed).
    color is BGR (OpenCV convention).
    """
    if not text:
        return frame

    font = _load_font(lang_code)

    if font is None or lang_code == "en":
        # Latin — OpenCV is fine
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (color[2], color[1], color[0]),   # BGR → BGR (cv2 uses BGR)
                    2, cv2.LINE_AA)
        return frame

    img_pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(img_pil)
    rgb_color = (color[2], color[1], color[0])   # BGR → RGB for Pillow
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# TRANSLATION HELPER
# ─────────────────────────────────────────────
def translate_word(word: str, lang_code: str) -> str:
    """
    Translate an English gesture label into the target language.
    For Odia: first checks the hand-curated get_odia() dict (offline, instant).
    For English: returns the word unchanged.
    Falls back to English on any API error.
    """
    if lang_code == "en":
        return word

    if lang_code == "or":
        # Use the existing curated Odia dict from scripts/odia_labels.py
        odia = get_odia(word)
        if odia and odia != word:   # get_odia returns the word itself when not found
            return odia

    try:
        result = translator.translate(word, dest=lang_code)
        return result.text
    except Exception as exc:
        print(f"[TRANSLATE ERROR] {exc}")
        return word   # graceful degradation


# ─────────────────────────────────────────────
# DRAW LANDMARKS  (identical to original infer.py)
# ─────────────────────────────────────────────
def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style()
    )


# ─────────────────────────────────────────────
# INFERENCE ENGINE  (identical logic to original infer.py)
#
# CRITICAL: to_relative_holistic() is called here on the
# FULL (30, 1662) sequence — NOT inside the per-frame extraction loop.
# Calling it on a single (1662,) vector causes the IndexError.
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(frame_buffer, prob_buffer):
    """
    Run GRU on the current 30-frame buffer.

    Pipeline (matches training exactly):
      1. np.array(frame_buffer)  → shape (30, 1662)   raw landmark sequences
      2. to_relative_holistic()  → shape (30, 1662)   body-relative normalisation
      3. unsqueeze(0)            → shape (1, 30, 1662) batch dim for model
      4. softmax + prob averaging for stable output
    """
    seq = np.array(frame_buffer, dtype=np.float32)   # (30, 1662) — raw
    seq = to_relative_holistic(seq)                   # (30, 1662) — normalised

    x     = torch.tensor(seq).unsqueeze(0).to(DEVICE) # (1, 30, 1662)
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
#   TOP BANNER — language selector + key hints
#   TOP BOX    — English word + confidence
#   BOTTOM BOX — Translated text (Unicode via Pillow)
#   Bars       — confidence bar + buffer fill bar
# ─────────────────────────────────────────────
def draw_hud(frame, prediction, confidence, buf_len,
             selected_language, translated_text, lang_code):
    h, w = frame.shape[:2]

    # ── Language banner (very top) ────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Lang: {selected_language}   Keys: 1=En 2=Hi 3=Od 4=Ta 5=Bn 6=Te   q=Quit",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA,
    )

    # ── English prediction box ────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 32), (w, 102), (0, 0, 0), -1)
    if prediction:
        cv2.putText(
            frame,
            f"{prediction}  ({confidence*100:.1f}%)",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 80), 3
        )
    else:
        cv2.putText(
            frame, "...", (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150, 150, 150), 2
        )

    # ── Translated text box (bottom) ──────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
    if prediction and translated_text:
        frame = put_unicode_text(
            frame, translated_text,
            (10, h - 90),
            lang_code,
            color=(0, 255, 255),   # cyan in BGR
        )

    # ── Confidence bar ────────────────────────────────────────────────────────
    conf_w    = int(confidence * (w - 20))
    bar_color = (0, 200, 0) if confidence >= CONF_THRESH else (0, 0, 200)
    cv2.rectangle(frame, (10, h - 25), (10 + conf_w, h - 17), bar_color, -1)
    cv2.rectangle(frame, (10, h - 25), (w - 10,      h - 17), (80, 80, 80), 1)

    # ── Buffer fill bar (orange) ──────────────────────────────────────────────
    fill_w = int((buf_len / SEQ_LEN) * (w - 20))
    cv2.rectangle(frame, (10, h - 10), (10 + fill_w, h - 3), (255, 160, 0), -1)
    cv2.rectangle(frame, (10, h - 10), (w - 10,      h - 3), (80, 80, 80), 1)

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

    # ── State (identical to original infer.py) ────────────────────────────────
    frame_buffer = deque(maxlen=SEQ_LEN)
    prob_buffer  = deque(maxlen=SMOOTH_WINDOW)
    frame_count  = 0
    debounce     = 0
    last_pred    = None
    confidence   = 0.0

    # ── Multilingual state ────────────────────────────────────────────────────
    selected_language = "Hindi"          # default on startup
    translated_text   = ""
    last_translated_word = ""            # avoid re-translating the same word

    print("📷 Camera running")
    print("   Keys: 1=English  2=Hindi  3=Odia  4=Tamil  5=Bengali  6=Telugu  q=Quit\n")

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
            frame = cv2.flip(frame, 1)   # mirror — natural webcam view

            # ── MediaPipe ─────────────────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            draw_landmarks(frame, results)

            # ── Feature extraction (raw — NO normalisation here) ──────────────
            # extract_holistic_landmarks() returns a flat (1662,) numpy array.
            # Normalisation (to_relative_holistic) happens inside predict()
            # on the full (30, 1662) batch — exactly as in training / original infer.py.
            landmarks = extract_holistic_landmarks(results)
            if landmarks.shape[0] == FEAT_SIZE_HOLISTIC:
                frame_buffer.append(landmarks)

            # ── Gate: only run model when hands are visible ───────────────────
            hands_detected = (
                results.left_hand_landmarks  is not None or
                results.right_hand_landmarks is not None
            )

            if (len(frame_buffer) == SEQ_LEN and
                    frame_count % PRED_EVERY_N == 0 and
                    hands_detected):

                last_pred, confidence = predict(frame_buffer, prob_buffer)

                # ── Translate whenever prediction changes ─────────────────────
                if last_pred and last_pred != last_translated_word:
                    lang_code        = LANGUAGES[selected_language]
                    translated_text  = translate_word(last_pred, lang_code)
                    last_translated_word = last_pred

                debounce += 1
                if last_pred and debounce >= DEBOUNCE_FRAMES:
                    print(f"[Detected] {last_pred} → {translated_text}  ({confidence*100:.1f}%)")
                    debounce = 0

            elif not hands_detected:
                last_pred    = None
                confidence   = 0.0
                prob_buffer.clear()

            # ── HUD ───────────────────────────────────────────────────────────
            lang_code = LANGUAGES[selected_language]
            frame = draw_hud(
                frame, last_pred, confidence, len(frame_buffer),
                selected_language, translated_text, lang_code,
            )

            cv2.imshow("ISL — Multilingual Real-Time Detection", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key in KEY_MAP:
                selected_language    = KEY_MAP[key]
                last_translated_word = ""   # force re-translation on next prediction
                translated_text      = ""
                print(f"[LANG] Switched to {selected_language}")
                # Pre-load font for the newly selected language
                _load_font(LANGUAGES[selected_language])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_inference()