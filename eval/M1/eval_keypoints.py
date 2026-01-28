# M1/extract_predictions_and_gt.py
from ultralytics import YOLO
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2


MODEL_PATH = "best.pt"
DATASET_ROOT = Path("")
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test" / "labels"
OUTPUT_DIR = Path("")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


test_images = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")) +
                     list(TEST_IMAGES_DIR.glob("*.png")))

if len(test_images) == 0:
    print(f"   Please check that the path is correct.")
    exit(1)

print(f"Found {len(test_images)} test images")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")


all_predictions = []
all_ground_truths = []

print(f"\n Processing {len(test_images)} images...")

for img_path in tqdm(test_images, desc="Extracting"):
    # ------------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------------
    results = model(img_path, verbose=False)[0]

    # Bild-Dimensionen
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]

    if results.keypoints is not None and len(results.keypoints) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        keypoints = results.keypoints.xy.cpu().numpy()  # (N_detections, 5, 2)
        kp_conf = results.keypoints.conf.cpu().numpy()  # (N_detections, 5)
        box_conf = results.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            pred = {
                'image': img_path.name,
                'image_width': img_width,
                'image_height': img_height,
                'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2] in pixels
                'bbox_confidence': float(box_conf[i]),
                'keypoints': []
            }

            # Keypoints
            for j in range(5):
                pred['keypoints'].append({
                    'x': float(keypoints[i][j][0]),  # Pixel coordinates
                    'y': float(keypoints[i][j][1]),
                    'confidence': float(kp_conf[i][j])
                })

            all_predictions.append(pred)


    # GROUND TRUTH
    label_path = TEST_LABELS_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"Warning: No label file for {img_path.name}")
        continue

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()

            # YOLO Format: class_id x_center y_center width height kp1_x kp1_y kp1_vis ...
            if len(parts) < 20:  # 5 bbox params + 5*3 keypoint params
                continue

            # Parse Bounding Box (normalized coordinates)
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # Konvertiere zu Pixel-Koordinaten
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height

            # Konvertiere zu [x1, y1, x2, y2]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            gt = {
                'image': img_path.name,
                'image_width': img_width,
                'image_height': img_height,
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],  # Pixel coordinates
                'bbox_normalized': [x_center_norm, y_center_norm, width_norm, height_norm],
                'keypoints': []
            }

            # Parse Keypoints (normalized coordinates)
            for i in range(5):
                idx = 5 + i * 3
                x_norm = float(parts[idx])
                y_norm = float(parts[idx + 1])
                visibility = float(parts[idx + 2])

                # Konvertiere zu Pixel-Koordinaten
                x_pixel = x_norm * img_width
                y_pixel = y_norm * img_height

                gt['keypoints'].append({
                    'x': x_pixel,  # Pixel coordinates
                    'y': y_pixel,
                    'x_normalized': x_norm,
                    'y_normalized': y_norm,
                    'visibility': visibility
                })

            all_ground_truths.append(gt)

predictions_file = OUTPUT_DIR / 'predictions.json'
ground_truth_file = OUTPUT_DIR / 'ground_truth.json'

print(f"\nSaving results...")

with open(predictions_file, 'w') as f:
    json.dump(all_predictions, f, indent=2)

with open(ground_truth_file, 'w') as f:
    json.dump(all_ground_truths, f, indent=2)

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print(f"Statistics:")
print(f"   Total images processed  : {len(test_images)}")
print(f"   Total predictions       : {len(all_predictions)}")
print(f"   Total ground truths     : {len(all_ground_truths)}")
print(f"\n Output files:")
print(f"   Predictions  : {predictions_file}")
print(f"   Ground Truth : {ground_truth_file}")
print("=" * 80)

# Validierung
if len(all_predictions) == 0:
    print("  WARNING: No predictions were extracted!")
    print("   Check that your model is working correctly.")

if len(all_ground_truths) == 0:
    print("  WARNING: No ground truths were extracted!")
    print("   Check that label files exist in:", TEST_LABELS_DIR)

if abs(len(all_predictions) - len(all_ground_truths)) > len(test_images):
    print(f"  WARNING: Mismatch between predictions ({len(all_predictions)}) "
          f"and ground truths ({len(all_ground_truths)})")
    print("   This might indicate detection failures or labeling issues.")