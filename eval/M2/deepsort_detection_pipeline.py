"""
DeepSORT Tracker für YOLO-Pose Detections
Benötigt: pip install deep-sort-realtime
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List
import cv2
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

ID_SWAP_DIST_THRESH = 50  # Pixel

# ----------------------------- Configuration ---------------------------------------
MAX_AGE = 70  # Frames ohne Update bevor Track gelöscht wird
N_INIT = 3  # Frames für Track-Bestätigung
MAX_IOU_DISTANCE = 0.7  # Maximum IoU distance für matching
MAX_COSINE_DISTANCE = 0.3  # Maximum cosine distance für appearance
NN_BUDGET = 100  # Feature-History Größe
EMBEDDER = 'mobilenet'  # 'mobilenet', 'torchreid', or 'clip'
MAX_ANIMALS = 11  # Maximum gleichzeitige Tiere

# Custom Feature Extraction (falls kein Video)
USE_CUSTOM_FEATURES = True
HIST_BINS = [16, 16, 16]


# ------------------------------------------------------------------------------------

def load_yolo_json(path: str):
    """Lade YOLO Detection JSON"""
    with open(path, 'r') as f:
        data = json.load(f)

    frames = {}
    if isinstance(data, list):
        for entry in data:
            frames[int(entry['frame'])] = entry['detections']
    elif isinstance(data, dict):
        for k, v in data.items():
            frames[int(k)] = v

    print(f" {len(frames)} Frames geladen")
    return frames


def extract_custom_features(image, bbox):
    """Extrahiere HSV Histogram als Fallback"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(512, dtype=np.float32)  # DeepSORT erwartet 512-dim

    crop = image[y1:y2, x1:x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Multi-scale histograms
    hist_h = cv2.calcHist([crop_hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([crop_hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([crop_hsv], [2], None, [16], [0, 256])

    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = cv2.normalize(hist, hist).flatten()

    # Pad to 512
    if len(hist) < 512:
        hist = np.pad(hist, (0, 512 - len(hist)), 'constant')

    return hist[:512].astype(np.float32)


def prepare_detections(detections_raw, frame_image=None):
    """
    Konvertiere YOLO Detections zu DeepSORT Format
    Format: [[bbox, confidence, class_id, feature], ...]
    bbox: [left, top, width, height]
    """
    detections = []

    for det in detections_raw:
        # YOLO bbox: [x1, y1, x2, y2] -> DeepSORT: [left, top, width, height]
        x1, y1, x2, y2 = det['bbox']
        left = x1
        top = y1
        width = x2 - x1
        height = y2 - y1

        # Confidence (falls vorhanden, sonst 1.0)
        confidence = det.get('confidence', 1.0)

        # Class (alle Tiere = Klasse 0)
        class_id = 0

        # Feature embedding
        feature = None
        if USE_CUSTOM_FEATURES and frame_image is not None:
            feature = extract_custom_features(frame_image, det['bbox'])

        class_id = 0  # alle Tiere = gleiche Klasse

        detection = ([left, top, width, height],
                     confidence,
                     class_id,
                     feature)

        detections.append(detection)

    return detections


def save_tracks_json(out_path: str, tracks_by_frame: Dict[int, List[dict]]):
    """Speichere Tracks als JSON"""
    with open(out_path, 'w') as f:
        json.dump(tracks_by_frame, f, indent=2)
    print(f" JSON gespeichert: {out_path}")


def save_tracks_csv(out_path: str, tracks_by_frame: Dict[int, List[dict]]):
    """Speichere Tracks als CSV"""
    import csv
    keys = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'confidence']

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)

        for frame, tracks in sorted(tracks_by_frame.items()):
            for t in tracks:
                bbox = t['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                conf = t.get('confidence', 1.0)
                writer.writerow([frame, t['track_id'],
                                 bbox[0], bbox[1], bbox[2], bbox[3],
                                 cx, cy, conf])

    print(f" CSV gespeichert: {out_path}")


def run_deepsort_tracking(yolo_json_path: str, video_path: str,
                          out_json: str, out_csv: str):
    """
    Hauptfunktion: DeepSORT Tracking
    """
    print("\n" + "=" * 70)
    print(" DeepSORT Multi-Animal Tracking")
    print("=" * 70 + "\n")

    # Load detections
    frames_data = load_yolo_json(yolo_json_path)

    # Initialize DeepSORT
    print(f"🔧 Initialisiere DeepSORT (embedder={EMBEDDER})...")

    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_iou_distance=MAX_IOU_DISTANCE,
        max_cosine_distance=MAX_COSINE_DISTANCE,
        nn_budget=NN_BUDGET,
        embedder=EMBEDDER,
        embedder_gpu=True,  # ✅ GPU aktivieren
        half=True,  # optional: FP16 (schneller)
        bgr=True,
        polygon=False
    )

    id_swaps = 0
    prev_active_tracks = {}  # track_id -> (cx, cy)
    fps = 25  # Fallback

    print(f" DeepSORT bereit!")

    # Open video if available
    cap = None
    use_video = False
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            use_video = True
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f" Video geladen: {video_path}")
            print(f"   FPS: {fps:.1f}, Frames: {total_frames}")
        else:
            print(f"  Video konnte nicht geladen werden, verwende Custom Features")
    else:
        print(f"  Kein Video angegeben, verwende Custom Features")

    # Storage
    tracks_by_frame = defaultdict(list)
    current_video_frame = 0
    all_frame_indices = sorted(frames_data.keys())

    print(f"\n Verarbeite {len(all_frame_indices)} Frames...\n")

    # Process frames
    for frame_idx in tqdm(all_frame_indices, desc="DeepSORT Tracking"):
        detections_raw = frames_data[frame_idx]

        # Read frame from video
        frame_image = None
        if use_video:
            while current_video_frame < frame_idx:
                cap.read()
                current_video_frame += 1

            ret, frame_image = cap.read()
            current_video_frame += 1

            if not ret:
                frame_image = None

        # Prepare detections for DeepSORT
        detections = prepare_detections(detections_raw, frame_image)

        # Update tracker
        if frame_image is not None:
            # DeepSORT mit Bild (verwendet eigenen Embedder)
            tracks = tracker.update_tracks(detections, frame=frame_image)
        else:
            # Ohne Bild (nur Custom Features)
            tracks = tracker.update_tracks(detections, frame=None)

        # Extract confirmed tracks
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()  # [left, top, right, bottom]

            # Finde ursprüngliche Detection für Keypoints
            original_det = None
            if hasattr(track, 'det_conf') and track.det_conf is not None:
                # Finde beste Match basierend auf bbox
                best_iou = 0
                for det in detections_raw:
                    x1, y1, x2, y2 = det['bbox']
                    det_bbox = [x1, y1, x2, y2]

                    # Simple IoU check
                    xi1 = max(bbox[0], det_bbox[0])
                    yi1 = max(bbox[1], det_bbox[1])
                    xi2 = min(bbox[2], det_bbox[2])
                    yi2 = min(bbox[3], det_bbox[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                    area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area2 = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        original_det = det

            keypoints = original_det.get('keypoints', []) if original_det else []
            confidence = original_det.get('confidence', 1.0) if original_det else 1.0

            active_tracks.append({
                "track_id": int(track_id),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "keypoints": keypoints,
                "confidence": float(confidence)
            })

        tracks_by_frame[frame_idx] = active_tracks
        # ---------------- ID SWAP DETECTION ----------------
        current_tracks = {}

        for t in active_tracks:
            bbox = t["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            current_tracks[t["track_id"]] = (cx, cy)

        # Check verschwundene Tracks
        for prev_id, (px, py) in prev_active_tracks.items():
            if prev_id in current_tracks:
                continue

            # Suche neuen Track in räumlicher Nähe
            for curr_id, (cx, cy) in current_tracks.items():
                dist = np.hypot(px - cx, py - cy)
                if dist < ID_SWAP_DIST_THRESH:
                    id_swaps += 1
                    if False:  # DEBUG
                        print(f"ID SWAP: {prev_id} -> {curr_id} @ frame {frame_idx}")
                    break

        prev_active_tracks = current_tracks.copy()
        # ---------------------------------------------------

    if cap:
        cap.release()

    # Statistics
    print(f"\n{'=' * 70}")
    print(" Tracking-Statistik")
    print(f"{'=' * 70}")

    total_tracks = 0
    frames_with_tracks = 0
    unique_ids = set()

    for frame, tracks in tracks_by_frame.items():
        if len(tracks) > 0:
            frames_with_tracks += 1
            total_tracks += len(tracks)
            for t in tracks:
                unique_ids.add(t['track_id'])

    print(f"Frames mit Tracks: {frames_with_tracks}/{len(all_frame_indices)}")
    print(f"Eindeutige Track-IDs: {len(unique_ids)}")
    print(f"Durchschnittliche Tracks/Frame: {total_tracks / len(all_frame_indices):.2f}")
    print(f"Max IDs (sollte ~{MAX_ANIMALS} sein): {max(unique_ids) if unique_ids else 0}")

    total_frames = len(all_frame_indices)
    total_minutes = total_frames / fps / 60.0

    id_swaps_per_minute = id_swaps / max(total_minutes, 1e-6)

    print(f"\n ID Swaps total: {id_swaps}")
    print(f"  ID Swaps / Minute: {id_swaps_per_minute:.2f}")

    # Save results
    save_tracks_json(out_json, tracks_by_frame)
    save_tracks_csv(out_csv, tracks_by_frame)

    print(f"\n{'=' * 70}\n")


# ----------------------------- CLI --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="DeepSORT Tracker für YOLO-Pose Detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Mit Video (verwendet DeepSORT's eigene Features):
  python deepsort_tracker.py --yolo-json predictions.json --video output.mp4

  # Ohne Video (verwendet Custom Color Features):
  python deepsort_tracker.py --yolo-json predictions.json

  # Custom Output-Pfade:
  python deepsort_tracker.py --yolo-json predictions.json --video output.mp4 \\
      --out-json deepsort_tracks.json --out-csv deepsort_tracks.csv
        """
    )

    p.add_argument('--yolo-json', required=True,
                   help='YOLO Detections JSON')
    p.add_argument('--video', default=None,
                   help='Video-Datei (optional, aber empfohlen)')
    p.add_argument('--out-json', default='deepsort_tracks.json',
                   help='Output JSON')
    p.add_argument('--out-csv', default='deepsort_tracks.csv',
                   help='Output CSV')
    p.add_argument('--embedder', default='mobilenet',
                   choices=['mobilenet', 'torchreid', 'clip'],
                   help='Feature Extractor (mobilenet=schnell, torchreid=genau)')
    p.add_argument('--max-age', type=int, default=70,
                   help='Max frames ohne Detection')
    p.add_argument('--n-init', type=int, default=3,
                   help='Frames für Track-Bestätigung')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Update globals with CLI args
    MAX_AGE = args.max_age
    N_INIT = args.n_init
    EMBEDDER = args.embedder

    run_deepsort_tracking(
        args.yolo_json,
        args.video,
        args.out_json,
        args.out_csv
    )