"""
Benutzung: python improved_detection_PipelineC.py --yolo-json detections.json --video input.mp4 --out-json outS.json --out-csv out.csv
"""

import argparse
import json
import os
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ----------------------------- Config ------------------------------------
MAX_ANIMALS = 11
MOTION_WEIGHT = 0.4
APPEARANCE_WEIGHT = 0.4
IOU_WEIGHT = 0.2

MOTION_DISTANCE_THRESHOLD = 150
IOU_THRESHOLD = 0.1
APPEARANCE_THRESHOLD = 0.6

MAX_AGE = 120 
MIN_HITS = 3
TENTATIVE_MAX_AGE = 15
CONFIRMED_TO_LOST_THRESHOLD = 30  

REIDENTIFY_ENABLED = True
REIDENTIFY_WINDOW = 1800  
REIDENTIFY_APPEARANCE_THRESHOLD = 0.58  
REIDENTIFY_SPATIAL_WEIGHT = 0.35  

GALLERY_SIZE = 20  # Beste N Features pro Track

HIST_BINS = [16, 16, 16]
USE_POSE_FEATURES = True
DEBUG = False


# ----------------------------- Data Classes ------------------------------
@dataclass
class Detection:
    frame: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    keypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    color_feature: Optional[np.ndarray] = None
    pose_feature: Optional[np.ndarray] = None


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter
    color_feature: np.ndarray
    pose_feature: Optional[np.ndarray]
    bbox: Tuple[float, float, float, float]
    keypoints: List[Tuple[float, float, float]]
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    hit_streak: int = 0
    state: str = "tentative"  # tentative, confirmed, lost
    history: deque = field(default_factory=lambda: deque(maxlen=500))
    feature_history: deque = field(default_factory=lambda: deque(maxlen=60))
    feature_gallery: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # NEU
    last_seen_frame: int = 0
    last_velocity: Optional[np.ndarray] = None  # NEU

    def predict(self):
        self.kf.predict()
        cx, cy = float(self.kf.x[0]), float(self.kf.x[1])
        return cx, cy

    def update(self, detection: Detection):
        cx, cy = detection_centroid(detection.bbox)

        # Speichere Velocity vor Update
        if self.kf.x is not None:
            self.last_velocity = self.kf.x[2:4].copy()

        self.kf.update(np.array([cx, cy]))

        # EMA für Color Feature (stabil)
        alpha = 0.15
        if detection.color_feature is not None:
            if self.color_feature is None or len(self.color_feature) == 0:
                self.color_feature = detection.color_feature.copy()
            else:
                self.color_feature = (1 - alpha) * self.color_feature + alpha * detection.color_feature

            # Gallery Update - speichere beste Features
            self._update_gallery(detection.color_feature, detection.confidence)

        if detection.pose_feature is not None:
            if self.pose_feature is None:
                self.pose_feature = detection.pose_feature.copy()
            else:
                self.pose_feature = (1 - alpha) * self.pose_feature + alpha * detection.pose_feature

        self.bbox = detection.bbox
        self.keypoints = detection.keypoints
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.age += 1
        self.history.append((detection.frame, self.bbox))
        if detection.color_feature is not None:
            self.feature_history.append(detection.color_feature.copy())
        self.last_seen_frame = detection.frame

        if self.state == "tentative" and self.hits >= MIN_HITS:
            self.state = "confirmed"
        elif self.state == "lost" and self.hit_streak > 0:
            self.state = "confirmed"

    def _update_gallery(self, feature: np.ndarray, confidence: float):
        """Speichere die besten N Features für robuste Re-ID"""
        self.feature_gallery.append((feature.copy(), confidence))
        # Behalte nur die besten Features
        self.feature_gallery.sort(key=lambda x: x[1], reverse=True)
        self.feature_gallery = self.feature_gallery[:GALLERY_SIZE]

    def get_gallery_features(self):
        """Gibt die gespeicherten Features zurück"""
        return [f[0] for f in self.feature_gallery]


# ----------------------------- Utilities --------------------------------

def detection_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, (x2_1 - x1_1)) * max(0, (y2_1 - y1_1))
    box2_area = max(0, (x2_2 - x1_2)) * max(0, (y2_2 - y1_2))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def create_kalman_filter(cx, cy, dt=1.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R *= 10.
    kf.Q = np.eye(4) * 0.01
    kf.x = np.array([cx, cy, 0., 0.])
    return kf


def extract_color_feature(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(int(np.prod(HIST_BINS)), dtype=np.float32)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(int(np.prod(HIST_BINS)), dtype=np.float32)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([crop_hsv], [0, 1, 2], None, HIST_BINS, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_pose_feature(keypoints):
    if len(keypoints) < 3:
        return None
    valid_kps = [kp for kp in keypoints if len(kp) >= 3 and kp[2] > 0.5]
    if len(valid_kps) < 3:
        return None
    points = np.array([[kp[0], kp[1]] for kp in valid_kps])
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(points[i] - points[j]))
    distances = np.array(distances)
    if distances.size == 0:
        return None
    if distances.max() > 0:
        distances = distances / distances.max()
    return distances.astype(np.float32)


def nms(detections: List[Detection], iou_thresh=0.5):
    if len(detections) == 0:
        return []
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    x1 = boxes[:, 0];
    y1 = boxes[:, 1];
    x2 = boxes[:, 2];
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]


def top_k_by_confidence(detections: List[Detection], k: int):
    if len(detections) <= k:
        return detections
    sorted_det = sorted(detections, key=lambda d: d.confidence, reverse=True)
    return sorted_det[:k]


# ----------------------------- Cost & Association ------------------------

def compute_cost_matrices(tracks: List[Track], detections: List[Detection]):
    n_tracks = len(tracks)
    n_dets = len(detections)
    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets)), np.zeros((n_tracks, n_dets)), np.zeros((n_tracks, n_dets))
    motion_cost = np.zeros((n_tracks, n_dets))
    appearance_cost = np.zeros((n_tracks, n_dets))
    iou_cost = np.zeros((n_tracks, n_dets))
    for i, tr in enumerate(tracks):
        pred_x, pred_y = float(tr.kf.x[0]), float(tr.kf.x[1])
        for j, det in enumerate(detections):
            det_x, det_y = detection_centroid(det.bbox)
            d = math.hypot(pred_x - det_x, pred_y - det_y)
            motion_cost[i, j] = min(d, MOTION_DISTANCE_THRESHOLD) / MOTION_DISTANCE_THRESHOLD
            try:
                color_dist = distance.cosine(tr.color_feature, det.color_feature)
            except Exception:
                color_dist = 1.0
            if np.isnan(color_dist):
                color_dist = 1.0
            app_cost = color_dist
            if USE_POSE_FEATURES and tr.pose_feature is not None and det.pose_feature is not None:
                if len(tr.pose_feature) == len(det.pose_feature):
                    pose_dist = np.linalg.norm(tr.pose_feature - det.pose_feature)
                    app_cost = 0.7 * color_dist + 0.3 * min(pose_dist, 1.0)
            appearance_cost[i, j] = min(max(app_cost, 0.0), 1.0)
            iou = bbox_iou(tr.bbox, det.bbox)
            iou_cost[i, j] = 1.0 - iou
    return motion_cost, appearance_cost, iou_cost


def associate_detections_to_tracks(tracks: List[Track], detections: List[Detection], threshold=0.7):
    if len(tracks) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
    motion_cost, appearance_cost, iou_cost = compute_cost_matrices(tracks, detections)
    combined = (MOTION_WEIGHT * motion_cost + APPEARANCE_WEIGHT * appearance_cost + IOU_WEIGHT * iou_cost)
    for i in range(len(tracks)):
        for j in range(len(detections)):
            if motion_cost[i, j] > 0.8 or appearance_cost[i, j] > 0.8:
                combined[i, j] = 999
    row_ind, col_ind = linear_sum_assignment(combined)
    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))
    for r, c in zip(row_ind, col_ind):
        if combined[r, c] <= threshold:
            matches.append((r, c))
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_detections:
                unmatched_detections.remove(c)
    return matches, unmatched_tracks, unmatched_detections


# ----------------------------- Re-ID ------------------------------------

def try_reidentify_improved(lost_tracks: List[Track], detections: List[Detection], current_frame: int):
    """Verbesserte Re-ID mit Spatial Priors und Gallery"""
    if not REIDENTIFY_ENABLED or len(lost_tracks) == 0 or len(detections) == 0:
        return []

    reid_matches = []
    cost_matrix = np.zeros((len(lost_tracks), len(detections)))

    for i, tr in enumerate(lost_tracks):
        frames_lost = current_frame - tr.last_seen_frame

        if len(tr.history) == 0 or frames_lost > REIDENTIFY_WINDOW:
            cost_matrix[i, :] = 999
            continue

        # Berechne erwartete Position
        expected_pos = None
        if len(tr.history) > 0 and tr.last_velocity is not None:
            last_frame, last_bbox = tr.history[-1]
            last_cx, last_cy = detection_centroid(last_bbox)
            vx, vy = float(tr.last_velocity[0]), float(tr.last_velocity[1])
            expected_pos = (last_cx + vx * frames_lost, last_cy + vy * frames_lost)

        for j, det in enumerate(detections):
            # 1. Appearance Cost mit Gallery
            app_cost = 1.0
            gallery_features = tr.get_gallery_features()

            if len(gallery_features) > 0:
                # Vergleiche mit allen Gallery-Features
                dists = []
                for gf in gallery_features:
                    try:
                        d = distance.cosine(gf, det.color_feature)
                        if not np.isnan(d):
                            dists.append(d)
                    except:
                        pass

                if len(dists) > 0:
                    # Nutze Minimum und Median für Robustheit
                    app_cost = 0.6 * min(dists) + 0.4 * np.median(dists)
            else:
                # Fallback auf feature_history
                if len(tr.feature_history) > 0:
                    hist_features = list(tr.feature_history)
                    dists = []
                    for f in hist_features:
                        try:
                            d = distance.cosine(f, det.color_feature)
                            if not np.isnan(d):
                                dists.append(d)
                        except:
                            pass
                    app_cost = min(dists) if len(dists) > 0 else 1.0

            # 2. Spatial Cost
            spatial_cost = 0.0
            if expected_pos is not None:
                det_cx, det_cy = detection_centroid(det.bbox)
                spatial_dist = math.hypot(det_cx - expected_pos[0], det_cy - expected_pos[1])
                # Normalisiere (z.B. max 300 Pixel Bewegung als realistisch)
                spatial_cost = min(spatial_dist / 300.0, 1.0)

            # 3. Pose Cost (wenn verfügbar)
            pose_cost = 0.0
            if USE_POSE_FEATURES and tr.pose_feature is not None and det.pose_feature is not None:
                if len(tr.pose_feature) == len(det.pose_feature):
                    pose_dist = np.linalg.norm(tr.pose_feature - det.pose_feature)
                    pose_cost = min(pose_dist, 1.0)

            # Kombiniere Costs
            if USE_POSE_FEATURES and pose_cost > 0:
                combined = (0.5 * app_cost +
                            0.25 * spatial_cost +
                            0.25 * pose_cost)
            else:
                combined = (0.65 * app_cost + 0.35 * spatial_cost)

            cost_matrix[i, j] = combined

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < REIDENTIFY_APPEARANCE_THRESHOLD:
            reid_matches.append((r, c))

    return reid_matches


# ----------------------------- Tracker Klasse ---------------------------
class ImprovedTrackerA:
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Detection], frame_image=None):
        self.frame_count += 1
        # Enforce per-frame detection cap early (NMS + TopK)
        detections = nms(detections, iou_thresh=0.4)
        if len(detections) > MAX_ANIMALS:
            detections = top_k_by_confidence(detections, MAX_ANIMALS)

        active_tracks = [t for t in self.tracks if t.state in ["confirmed", "tentative"]]

        # Predict
        for tr in self.tracks:
            tr.predict()
            tr.time_since_update += 1
            tr.age += 1
            if tr.time_since_update > 0:
                tr.hit_streak = 0

        # 1) Associate confirmed
        confirmed = [t for t in self.tracks if t.state == "confirmed"]
        matches, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(confirmed, detections,
                                                                                         threshold=0.7)
        for tr_idx, det_idx in matches:
            tr = confirmed[tr_idx]
            det = detections[det_idx]
            tr.update(det)

        # 2) Tentative matching
        tentative = [t for t in self.tracks if t.state == "tentative"]
        if len(tentative) > 0 and len(unmatched_detections) > 0:
            remaining = [detections[i] for i in unmatched_detections]
            tent_matches, tent_un_tr, tent_un_det = associate_detections_to_tracks(tentative, remaining, threshold=0.8)
            matched_det_indices = set()
            for tr_idx, det_idx in tent_matches:
                if tr_idx < len(tentative) and det_idx < len(remaining):
                    tentative[tr_idx].update(remaining[det_idx])
                    matched_det_indices.add(unmatched_detections[det_idx])
            unmatched_detections = [i for i in unmatched_detections if i not in matched_det_indices]

        # 3) Verbesserte Re-ID mit Spatial Priors
        if len(self.lost_tracks) > 0 and len(unmatched_detections) > 0:
            remaining = [detections[i] for i in unmatched_detections]
            reid_matches = try_reidentify_improved(self.lost_tracks, remaining, self.frame_count)
            used_lost = []
            used_det_indices = set()
            for lost_idx, det_idx in reid_matches:
                if lost_idx >= len(self.lost_tracks) or det_idx >= len(remaining):
                    continue
                tr = self.lost_tracks[lost_idx]
                det = remaining[det_idx]
                tr.update(det)
                tr.state = "confirmed"
                tr.time_since_update = 0
                self.tracks.append(tr)
                used_lost.append(tr)
                used_det_indices.add(unmatched_detections[det_idx])
                if DEBUG:
                    print(f"Re-ID: restored {tr.track_id} after {self.frame_count - tr.last_seen_frame} frames")
            self.lost_tracks = [t for t in self.lost_tracks if t not in used_lost]
            unmatched_detections = [i for i in unmatched_detections if i not in used_det_indices]

        # 4) Create new tracks but enforce MAX_ANIMALS
        active_count = len([t for t in self.tracks if t.state in ["confirmed", "tentative"]])
        available_slots = max(0, MAX_ANIMALS - active_count)
        for idx in list(unmatched_detections):
            if available_slots <= 0:
                break
            det = detections[idx]
            cx, cy = detection_centroid(det.bbox)
            kf = create_kalman_filter(cx, cy)
            new_track = Track(
                track_id=self.next_id,
                kf=kf,
                color_feature=det.color_feature.copy() if det.color_feature is not None else np.zeros(
                    int(np.prod(HIST_BINS)), dtype=np.float32),
                pose_feature=det.pose_feature.copy() if det.pose_feature is not None else None,
                bbox=det.bbox,
                keypoints=det.keypoints,
                state="tentative"
            )
            new_track.update(det)
            self.tracks.append(new_track)
            self.next_id += 1
            available_slots -= 1

        # 5) Manage track lifecycles mit angepassten Schwellen
        tracks_to_remove = []
        for tr in list(self.tracks):
            if tr.time_since_update > self.max_age:
                tracks_to_remove.append(tr)
            elif tr.state == "tentative" and tr.time_since_update > TENTATIVE_MAX_AGE:
                tracks_to_remove.append(tr)
            elif tr.time_since_update > CONFIRMED_TO_LOST_THRESHOLD and tr.state == "confirmed":
                tr.state = "lost"
                tr.time_since_update = 0
                self.lost_tracks.append(tr)
                tracks_to_remove.append(tr)

        for tr in tracks_to_remove:
            if tr in self.tracks:
                self.tracks.remove(tr)

        # cleanup lost_tracks window
        cleaned_lost = []
        for t in self.lost_tracks:
            if len(t.history) == 0:
                continue
            if self.frame_count - t.last_seen_frame < REIDENTIFY_WINDOW:
                cleaned_lost.append(t)
        self.lost_tracks = cleaned_lost

        # return confirmed tracks
        active = []
        for tr in self.tracks:
            if tr.state == "confirmed":
                active.append({
                    "track_id": tr.track_id,
                    "bbox": list(map(float, tr.bbox)),
                    "keypoints": tr.keypoints,
                    "state": tr.state
                })
        return active


# ----------------------------- IO helpers & runner ----------------------

def load_yolo_json(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    frames = {}
    if isinstance(data, list):
        for entry in data:
            frames[int(entry['frame'])] = entry['detections']
    elif isinstance(data, dict):
        for k, v in data.items():
            frames[int(k)] = v
    return frames


def save_tracks_json(out_path: str, tracks_by_frame: Dict[int, List[dict]]):
    with open(out_path, 'w') as f:
        json.dump(tracks_by_frame, f, indent=2)


def save_tracks_csv(out_path: str, tracks_by_frame: Dict[int, List[dict]]):
    import csv
    keys = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'state']
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        for frame, tracks in sorted(tracks_by_frame.items()):
            for t in tracks:
                bbox = t['bbox']
                cx, cy = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                writer.writerow(
                    [frame, t['track_id'], bbox[0], bbox[1], bbox[2], bbox[3], cx, cy, t.get('state', 'confirmed')])


def run_pipeline(yolo_json_path: str, video_path: Optional[str], out_json: str, out_csv: str):
    frames = load_yolo_json(yolo_json_path)
    tracker = ImprovedTrackerA(max_age=MAX_AGE, min_hits=MIN_HITS)
    tracks_by_frame = defaultdict(list)

    cap = None
    use_video = video_path and os.path.exists(video_path)
    if use_video:
        cap = cv2.VideoCapture(video_path)

    current_video_frame = 0
    all_frame_indices = sorted(frames.keys())

    for frame_idx in tqdm(all_frame_indices, desc="Tracking"):
        detections_raw = frames[frame_idx]
        frame_image = None
        if use_video:
            while current_video_frame < frame_idx:
                ret_skip, _ = cap.read()
                current_video_frame += 1
                if not ret_skip and current_video_frame < frame_idx:
                    break
            ret, frame_image = cap.read()
            current_video_frame += 1
            if not ret:
                frame_image = None

        detections = []
        for d in detections_raw:
            bbox = tuple(d['bbox'])
            conf = float(d.get('confidence', 1.0))
            keypoints = d.get('keypoints', [])
            det = Detection(frame=frame_idx, bbox=bbox, confidence=conf, keypoints=keypoints)
            if frame_image is not None:
                det.color_feature = extract_color_feature(frame_image, bbox)
            else:
                det.color_feature = np.zeros(int(np.prod(HIST_BINS)), dtype=np.float32)
            if USE_POSE_FEATURES:
                det.pose_feature = extract_pose_feature(keypoints)
            else:
                det.pose_feature = None
            detections.append(det)

        active_tracks = tracker.update(detections, frame_image=frame_image)
        tracks_by_frame[frame_idx] = active_tracks

    save_tracks_json(out_json, tracks_by_frame)
    save_tracks_csv(out_csv, tracks_by_frame)
    if cap:
        cap.release()


# ----------------------------- CLI -------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--yolo-json', required=True)
    p.add_argument('--video', required=False, default=None)
    p.add_argument('--out-json', default='tracked_variantA_improved.json')
    p.add_argument('--out-csv', default='tracked_variantA_improved.csv')
    args = p.parse_args()
    run_pipeline(args.yolo_json, args.video, args.out_json, args.out_csv)