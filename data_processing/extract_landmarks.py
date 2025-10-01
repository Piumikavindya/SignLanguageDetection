import os
import cv2
import csv
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def normalize_landmarks(landmarks):
    arr = np.array(landmarks)
    arr = arr - arr[0]  # subtract wrist as origin
    max_norm = np.max(np.linalg.norm(arr, axis=1))
    if max_norm > 1e-6:
        arr = arr / max_norm
    return arr.flatten()

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    paired = list(zip(
        res.multi_hand_landmarks,
        [h.classification[0].label for h in res.multi_handedness]
    ))
    # Sort to keep Right first, Left second (if both exist)
    paired_sorted = sorted(paired, key=lambda x: 0 if x[1] == 'Right' else 1)
    
    features = []
    for lm, _ in paired_sorted[:2]:
        coords = [(p.x, p.y, p.z) for p in lm.landmark]
        features.extend(normalize_landmarks(coords))
    
    # If only one hand detected, pad the other with zeros
    if len(paired_sorted) == 1:
        features.extend([0.0] * 63)
    
    return features

def dump_dataset(root_folder, out_csv='landmarks.csv', log_file='skipped_images.txt'):
    rows = []
    skipped = []

    classes = sorted(os.listdir(root_folder))
    for cls in classes:
        cls_path = os.path.join(root_folder, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            p = os.path.join(cls_path, fname)
            feat = process_image(p)
            if feat is None:
                skipped.append(p)
                continue
            rows.append([cls] + feat)

    # Save valid landmarks
    if rows:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'f{i}' for i in range(len(rows[0]) - 1)]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"✅ Saved {len(rows)} samples to {out_csv}")
    else:
        print("⚠️ No valid samples found, CSV not created.")

    # Save skipped image log
    if skipped:
        with open(log_file, 'w') as f:
            for s in skipped:
                f.write(s + '\n')
        print(f"⚠️ Skipped {len(skipped)} images (no hands found). Logged in {log_file}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "DataSets", "images_for_phrases")
    dump_dataset(
        dataset_path,
        out_csv=os.path.join(base_dir, "data_processing", "landmarks.csv"),
        log_file=os.path.join(base_dir, "data_processing", "skipped_images.txt")
    )
