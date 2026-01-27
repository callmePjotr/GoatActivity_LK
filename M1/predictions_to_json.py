from ultralytics import YOLO
import json
import os
from tqdm import tqdm
import cv2

model_path = "path_to_best_model/best.pt"
video_path = "path_to_video/video.mp4"
output_json = "path_to_output_folder/output.json"

model = YOLO(model_path)

# Videolänge bestimmen
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# YOLO Prediction 
results = model.predict(source=video_path, save=True, stream=True, verbose=False)

frames = []
for frame_idx, result in enumerate(tqdm(results, total=total_frames, desc="Processing frames")):
    detections = []
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    kpts = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

    for i in range(len(boxes)):
        bbox = boxes[i].tolist()
        keypoints = []
        if len(kpts) > i:
            pts = kpts[i]
            for p in pts:
                if len(p) == 3:
                    keypoints.append([float(p[0]), float(p[1]), float(p[2])])
                else:
                    keypoints.append([float(p[0]), float(p[1]), 1.0])

        detections.append({
            "bbox": bbox,
            "confidence": float(confs[i]) if len(confs) > i else None,
            "keypoints": keypoints
        })

    frames.append({"frame": frame_idx, "detections": detections})

# speichern
with open(output_json, "w") as f:
    json.dump(frames, f, indent=2)

print(f"✅ YOLO Pose JSON exportiert nach: {output_json}")
