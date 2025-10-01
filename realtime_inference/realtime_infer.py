import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pickle

# ----------------------------------------------------
# Path handling (always relative to this script folder)
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # -> SignLanguageDetection/realtime_inference
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_training")

# ----------------------------------------------------
# Choose model type: 'mlp' or 'rf'
# ----------------------------------------------------
model_type = 'mlp'  # change to 'rf' if you want RandomForest

if model_type == 'mlp':
    model = load_model(os.path.join(MODEL_DIR, "mlp_landmarks.h5"))
else:
    model = joblib.load(os.path.join(MODEL_DIR, "rf_landmark_model.joblib"))

# Load Label Encoder
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# ----------------------------------------------------
# Mediapipe setup
# ----------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------------------------------
# Landmark processing helpers
# ----------------------------------------------------
def normalize_landmarks(landmarks):
    arr = np.array(landmarks)
    arr = arr - arr[0]  # normalize relative to wrist
    max_norm = np.max(np.linalg.norm(arr, axis=1))
    if max_norm > 1e-6:
        arr = arr / max_norm
    return arr.flatten()

def extract_landmarks_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return None

    paired = list(zip(
        res.multi_hand_landmarks,
        [h.classification[0].label for h in res.multi_handedness]
    ))

    # Ensure Right hand comes first, then Left
    paired_sorted = sorted(paired, key=lambda x: 0 if x[1] == 'Right' else 1)

    features = []
    for lm, _ in paired_sorted[:2]:
        coords = [(p.x, p.y, p.z) for p in lm.landmark]
        features.extend(normalize_landmarks(coords))

    # If only one hand detected, pad with zeros
    if len(paired_sorted) == 1:
        features.extend([0.0] * 63)

    return np.array(features).reshape(1, -1)

# ----------------------------------------------------
# Realtime loop
# ----------------------------------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks_from_frame(frame)

    if landmarks is not None:
        if model_type == 'mlp':
            pred = model.predict(landmarks)
            pred_class = le.inverse_transform([np.argmax(pred)])[0]
        else:
            pred_class = le.inverse_transform(model.predict(landmarks))[0]

        cv2.putText(frame, pred_class, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
