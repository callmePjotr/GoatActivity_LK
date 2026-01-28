import cv2
import os
import numpy as np
from pathlib import Path

TARGET_SIZE = 224
INPUT_BASE = r"L:\Uni\Master\Master Thesis\RE_ID_MODELL\YOLO_DATASET_4_PREDICTIONS\detection\track_crops_corrected\merged_activitys"
OUTPUT_BASE = r"L:\Uni\Master\Master Thesis\RE_ID_MODELL\YOLO_DATASET_4_PREDICTIONS\detection\track_crops_corrected\normalized_clips_full"

def letterbox(img, size=224):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    padded[y0:y0+nh, x0:x0+nw] = resized
    return padded

def process_video(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = letterbox(frame, TARGET_SIZE)
        frames.append(frame)

    cap.release()

    np.save(os.path.join(output_dir, "frames.npy"), np.array(frames))
    print(f"[OK] {input_path} → {len(frames)} Frames gespeichert.")

def run():
    for activity in os.listdir(INPUT_BASE):
        act_dir = os.path.join(INPUT_BASE, activity)
        if not os.path.isdir(act_dir):
            continue

        for filename in os.listdir(act_dir):
            if not filename.endswith(".mp4"):
                continue

            input_path = os.path.join(act_dir, filename)
            out_dir = os.path.join(OUTPUT_BASE, activity, filename.replace(".mp4", ""))
            process_video(input_path, out_dir)

run()
