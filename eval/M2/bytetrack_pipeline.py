import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT

except ImportError:
    print("installiere ultralytics")
    import sys

    sys.exit(1)

ID_SWAP_DIST_THRESH = 50



def create_tracker_yaml(tracker_type="bytetrack", output_dir="."):

    bytetrack_yaml = """# ByteTrack Tracker Configuration
    tracker_type: bytetrack

    # Thresholds
    track_high_thresh: 0.5
    track_low_thresh: 0.1
    new_track_thresh: 0.6

    # Matching
    match_thresh: 0.8
    fuse_score: false

    # Track management
    track_buffer: 70
    frame_rate: 30
    mot20: false
    """

    botsort_yaml = """# BoT-SORT Tracker Configuration
    tracker_type: botsort

    # Thresholds
    track_high_thresh: 0.5
    track_low_thresh: 0.1
    new_track_thresh: 0.6

    # Matching
    match_thresh: 0.8
    proximity_thresh: 0.5
    appearance_thresh: 0.25
    fuse_score: true

    # Track management
    track_buffer: 70
    frame_rate: 30
    mot20: false

    # Global Motion Compensation (REQUIRED)
    gmc_method: sparseOptFlow

    # ReID
    with_reid: true
    model: auto
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    bytetrack_path = output_path / "bytetrack.yaml"
    botsort_path = output_path / "botsort.yaml"

    with open(bytetrack_path, "w") as f:
        f.write(bytetrack_yaml)
    print(f" ByteTrack Config: {bytetrack_path}")

    with open(botsort_path, "w") as f:
        f.write(botsort_yaml)
    print(f" BoT-SORT Config: {botsort_path}")

    return str(bytetrack_path), str(botsort_path)


def load_detections_json(path):
    """Lädt Custom YOLO Detections aus JSON"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = {}
    if isinstance(data, list):
        for e in data:
            frames[int(e["frame"])] = e["detections"]
    else:
        for k, v in data.items():
            frames[int(k)] = v

    print(f" {len(frames)} Frames aus JSON geladen")
    return frames


def save_json(path, data):
    """Speichert Tracking-Ergebnisse als JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f" JSON gespeichert: {path}")


def save_csv(path, tracks_by_frame):
    """Speichert Tracking-Ergebnisse als CSV"""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "conf"])
        for frame, tracks in sorted(tracks_by_frame.items()):
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                conf = t.get("confidence", 1.0)
                writer.writerow([frame, t["track_id"], x1, y1, x2, y2, cx, cy, conf])
    print(f" CSV gespeichert: {path}")


def analyze_id_swaps(tracks_by_frame, dist_thresh=50):
    """Analysiert ID Swaps"""
    prev_tracks = {}
    id_swaps = 0

    for frame_idx in sorted(tracks_by_frame.keys()):
        current_tracks = {}

        for t in tracks_by_frame[frame_idx]:
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_tracks[t["track_id"]] = (cx, cy)

        for pid, (px, py) in prev_tracks.items():
            if pid in current_tracks:
                continue
            for cid, (cx, cy) in current_tracks.items():
                dist = math.hypot(px - cx, py - cy)
                if dist < dist_thresh:
                    id_swaps += 1
                    break

        prev_tracks = current_tracks

    return id_swaps


class ByteTrackerArgs:
    """Konfiguration für ByteTracker"""

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=70, fps=30):
        self.track_thresh = track_thresh
        self.track_high_thresh = track_thresh + 0.1
        self.track_low_thresh = 0.1
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = fps
        self.mot20 = False
        # FIX für veraltete Ultralytics Version
        self.fuse_score = False  # Fehlender Parameter


# ---------------- Tracking von Custom Detections ----------------
def run_tracking_from_detections(
        detections_json,
        tracker="bytetrack",
        output_dir="output",
        fps=30,
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=70
):
    """
    Trackt Custom YOLO Detections mit ByteTrack oder BoT-SORT

    Args:
        detections_json: JSON mit Custom YOLO Detections
        tracker: "bytetrack" oder "botsort"
        output_dir: Output-Verzeichnis
        fps: Video FPS
        track_thresh: Detection Threshold
        match_thresh: Matching Threshold
        track_buffer: Track Buffer (Frames)
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Lade Custom Detections
    print(f"\n Lade Custom YOLO Detections...")
    frames = load_detections_json(detections_json)

    if len(frames) == 0:
        print(" Keine Frames gefunden!")
        return

    # Erstelle Tracker Config
    print(f"\n️ Erstelle Tracker Konfiguration...")
    bytetrack_yaml, botsort_yaml = create_tracker_yaml(output_dir=output_dir)
    tracker_config = bytetrack_yaml if tracker == "bytetrack" else botsort_yaml

    # Initialisiere Tracker
    print(f"\n Initialisiere {tracker.upper()}...")
    print(f"   Track Threshold:  {track_thresh}")
    print(f"   Match Threshold:  {match_thresh}")
    print(f"   Track Buffer:     {track_buffer}")
    print(f"   FPS:              {fps}")

    if tracker == "bytetrack":
        args = ByteTrackerArgs(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fps=fps
        )
        tracker_obj = BYTETracker(args=args, frame_rate=fps)
    else:  # botsort
        args = ByteTrackerArgs(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fps=fps
        )
        tracker_obj = BOTSORT(args=args, frame_rate=fps)

    # Tracking durchführen
    tracks_by_frame = defaultdict(list)
    total_tracks = 0
    total_detections = 0

    print(f"\n Starte {tracker.upper()} Tracking...")

    for frame_idx in tqdm(sorted(frames.keys()), desc=tracker.upper()):
        dets = []
        for d in frames[frame_idx]:
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("confidence", 1.0)
            # Format: [x1, y1, x2, y2, conf, class_id]
            dets.append([x1, y1, x2, y2, conf, 0])

        total_detections += len(dets)

        if len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.array(dets, dtype=np.float32)

        # Update Tracker
        try:
            online_targets = tracker_obj.update(dets, None)
        except Exception as e:
            print(f"\n Fehler in Frame {frame_idx}: {e}")
            online_targets = []

        # Sammle Tracks
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            tid = t.track_id
            tracks_by_frame[frame_idx].append({
                "track_id": int(tid),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(t.score) if hasattr(t, 'score') else 1.0
            })
            total_tracks = max(total_tracks, int(tid))

    # Analysiere ID Swaps
    print(f"\n Analysiere ID Swaps...")
    id_swaps = analyze_id_swaps(tracks_by_frame, ID_SWAP_DIST_THRESH)

    # Statistiken
    total_minutes = len(frames) / fps / 60
    avg_dets = total_detections / len(frames) if len(frames) > 0 else 0

    print(f"\n" + "=" * 60)
    print(f" {tracker.upper()} Ergebnisse:")
    print(f"=" * 60)
    print(f"   Frames:                      {len(frames)}")
    print(f"   Total Detections:            {total_detections}")
    print(f"   Ø Detections/Frame:          {avg_dets:.1f}")
    print(f"   Unique Track IDs:            {total_tracks}")
    print(f"   ID Swaps:                    {id_swaps}")
    print(f"   ID Swaps/min:                {id_swaps / max(total_minutes, 1e-6):.2f}")
    print(f"   Video Dauer:                 {total_minutes:.2f} min")
    print(f"=" * 60)

    # Speichere Ergebnisse
    print(f"\n Speichere Ergebnisse...")
    json_name = Path(detections_json).stem

    json_path = output_path / f"{json_name}_{tracker}_tracks.json"
    csv_path = output_path / f"{json_name}_{tracker}_tracks.csv"

    save_json(json_path, {
        "source": str(detections_json),
        "tracker": tracker,
        "config": {
            "track_thresh": track_thresh,
            "match_thresh": match_thresh,
            "track_buffer": track_buffer,
            "fps": fps
        },
        "statistics": {
            "frames": len(frames),
            "total_detections": total_detections,
            "unique_tracks": total_tracks,
            "id_swaps": id_swaps,
            "id_swaps_per_minute": id_swaps / max(total_minutes, 1e-6)
        },
        "tracks": dict(tracks_by_frame)
    })

    save_csv(csv_path, tracks_by_frame)

    print(f"\n Tracking abgeschlossen!")
    print(f" Output: {output_path}")

    return tracks_by_frame


# ---------------- Tracking direkt mit Custom Model ----------------
def run_tracking_with_model(
        video_path,
        model_path,
        tracker="bytetrack",
        output_dir="output",
        conf_threshold=0.5,
        iou_threshold=0.7,
        device=None,
        save_video=True
):
    """
    Trackt Video direkt mit Custom YOLO Model

    Args:
        video_path: Pfad zum Input-Video
        model_path: Pfad zu deinem Custom YOLO Modell (.pt)
        tracker: "bytetrack" oder "botsort"
        output_dir: Output-Verzeichnis
        conf_threshold: Detection Confidence
        iou_threshold: IOU für NMS
        device: 'cpu', 'cuda', oder None
        save_video: Video mit Tracking speichern
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Video Info
    print(f"\n Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"   Auflösung: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Frames: {frame_count}")
    print(f"   Dauer: {frame_count / fps:.2f} sec")

    # Erstelle Tracker Config
    print(f"\n️ Erstelle Tracker Config...")
    bytetrack_yaml, botsort_yaml = create_tracker_yaml(output_dir=output_dir)
    tracker_config = bytetrack_yaml if tracker == "bytetrack" else botsort_yaml

    # Lade Custom Model
    print(f"\n🤖 Lade Custom YOLO Model: {model_path}")
    model = YOLO(model_path)

    # Tracking
    print(f"\n Starte {tracker.upper()} Tracking...")
    print(f"   Tracker: {tracker_config}")
    print(f"   Confidence: {conf_threshold}")
    print(f"   IOU: {iou_threshold}")
    print(f"   Device: {device or 'auto'}")

    results = model.track(
        source=video_path,
        tracker=tracker_config,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        stream=True,
        verbose=False
    )

    # Sammle Results
    tracks_by_frame = defaultdict(list)
    total_tracks = 0
    total_detections = 0

    for frame_idx, r in enumerate(tqdm(results, total=frame_count, desc=tracker.upper())):
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            confidences = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box
                tracks_by_frame[frame_idx].append({
                    "track_id": int(tid),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf)
                })
                total_tracks = max(total_tracks, int(tid))
                total_detections += 1

    # ID Swaps
    id_swaps = analyze_id_swaps(tracks_by_frame, ID_SWAP_DIST_THRESH)

    # Stats
    total_minutes = frame_count / fps / 60
    avg_dets = total_detections / len(tracks_by_frame) if tracks_by_frame else 0

    print(f"\n" + "=" * 60)
    print(f" {tracker.upper()} Ergebnisse:")
    print(f"=" * 60)
    print(f"   Frames:                      {len(tracks_by_frame)}")
    print(f"   Detections:                  {total_detections}")
    print(f"   Ø Detections/Frame:          {avg_dets:.1f}")
    print(f"   Unique Tracks:               {total_tracks}")
    print(f"   ID Swaps:                    {id_swaps}")
    print(f"   ID Swaps/min:                {id_swaps / max(total_minutes, 1e-6):.2f}")
    print(f"=" * 60)

    # Save
    video_name = Path(video_path).stem
    json_path = output_path / f"{video_name}_{tracker}_tracks.json"
    csv_path = output_path / f"{video_name}_{tracker}_tracks.csv"

    save_json(json_path, {
        "video": str(video_path),
        "model": str(model_path),
        "tracker": tracker,
        "statistics": {
            "frames": len(tracks_by_frame),
            "detections": total_detections,
            "tracks": total_tracks,
            "id_swaps": id_swaps
        },
        "tracks": dict(tracks_by_frame)
    })

    save_csv(csv_path, tracks_by_frame)

    if save_video:
        print(f"\n Erstelle Video mit Tracking...")
        model.track(
            source=video_path,
            tracker=tracker_config,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            save=True,
            project=str(output_path),
            name=f"{video_name}_{tracker}",
            exist_ok=True
        )

    print(f"\n Fertig! Output: {output_path}")
    return tracks_by_frame


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ByteTrack/BoT-SORT mit Custom YOLO Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", help="Video-Datei für direktes Tracking mit Custom Model")
    mode.add_argument("--detections-json", help="JSON mit Custom YOLO Detections")

    # Model (nur für --video Mode)
    parser.add_argument("--model", help="Pfad zu Custom YOLO Model (.pt) - nur mit --video")

    # Tracker
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"],
                        default="bytetrack", help="Tracker (default: bytetrack)")

    # Tracker params
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--track-buffer", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)

    # Detection params (nur für --video)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)

    # Output
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-video", action="store_true")

    args = parser.parse_args()

    try:
        if args.video:
            # Direktes Tracking mit Custom Model
            if not args.model:
                print(" --model erforderlich wenn --video verwendet wird")
                exit(1)

            run_tracking_with_model(
                video_path=args.video,
                model_path=args.model,
                tracker=args.tracker,
                output_dir=args.output_dir,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                device=args.device,
                save_video=not args.no_video
            )
        else:
            # Tracking von existierenden Detections
            run_tracking_from_detections(
                detections_json=args.detections_json,
                tracker=args.tracker,
                output_dir=args.output_dir,
                fps=args.fps,
                track_thresh=args.track_thresh,
                match_thresh=args.match_thresh,
                track_buffer=args.track_buffer
            )
    except Exception as e:
        print(f"\n Fehler: {e}")
        import traceback

        traceback.print_exc()
        exit(1)