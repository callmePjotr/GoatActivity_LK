import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np

# ---------------- default settings ----------------
CROP_PADDING = 10
CROP_MIN_SIZE = 50
CROP_PADDING_AUGMENTATION = True
CROP_PADDING_MODE = 'random'
OUT_VIDEO_CODEC = 'mp4v'
DEFAULT_FPS = 30


SKELETON = []  # basierend auf Anzahl der Keypoints


# ---------------------------------------------------------------------------


def flatten_keypoints(keypoints):
    """Flatten keypoints to [x, y, conf, x, y, conf, ...] format"""
    if not keypoints:
        return []

    # If already flat list of numbers
    if len(keypoints) > 0 and isinstance(keypoints[0], (int, float)):
        return keypoints

    # If nested list [[x, y, conf], [x, y, conf], ...]
    if len(keypoints) > 0 and isinstance(keypoints[0], (list, tuple)):
        flat = []
        for kp in keypoints:
            if len(kp) >= 2:
                x = float(kp[0])
                y = float(kp[1])
                conf = float(kp[2]) if len(kp) > 2 else 1.0
                flat.extend([x, y, conf])
        return flat

    return []


def load_tracks(json_path):
    """Load tracks with keypoints from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    tracks = defaultdict(list)

    if isinstance(data, dict):
        for frame_s, dets in data.items():
            frame = int(frame_s)
            for d in dets:
                tid = int(d['track_id'])
                bbox = tuple(map(float, d['bbox']))
                keypoints_raw = d.get('keypoints', [])
                keypoints = flatten_keypoints(keypoints_raw)
                tracks[tid].append((frame, bbox, keypoints))
    elif isinstance(data, list):
        for entry in data:
            frame = int(entry.get('frame', entry.get('frame_idx', 0)))
            dets = entry.get('detections', [])
            for d in dets:
                tid = int(d['track_id'])
                bbox = tuple(map(float, d['bbox']))
                keypoints_raw = d.get('keypoints', [])
                keypoints = flatten_keypoints(keypoints_raw)
                tracks[tid].append((frame, bbox, keypoints))
    else:
        raise ValueError("Unsupported JSON format")

    return tracks


def group_into_tracklets(track_data, gap_threshold=10):
    """Group track data into continuous tracklets"""
    if not track_data:
        return []

    track_data = sorted(track_data, key=lambda x: x[0])
    tracklets = []
    current = [track_data[0]]

    for i in range(1, len(track_data)):
        if track_data[i][0] - track_data[i - 1][0] <= gap_threshold:
            current.append(track_data[i])
        else:
            tracklets.append(current)
            current = [track_data[i]]
    tracklets.append(current)
    return tracklets


def extract_crop_from_frame(frame, bbox, padding, min_size):
    """Extract crop with padding"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1_crop = max(0, x1 - padding)
    y1_crop = max(0, y1 - padding)
    x2_crop = min(w, x2 + padding)
    y2_crop = min(h, y2 + padding)

    if (x2_crop - x1_crop) < min_size or (y2_crop - y1_crop) < min_size:
        return None, None

    crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    crop_offset = (x1_crop, y1_crop)  # offset für keypoint transformation
    return crop, crop_offset


def _create_augmented_canvas(h, w, reference_img, padding_mode):
    """Create augmented padding background"""
    mode = padding_mode
    if mode == 'random':
        mode = np.random.choice(['black', 'blur', 'noise', 'edge'])

    if mode == 'black':
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif mode == 'blur':
        blurred = cv2.GaussianBlur(reference_img, (21, 21), 0)
        canvas = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        return (canvas * 0.3).astype(np.uint8)
    elif mode == 'noise':
        return np.random.normal(20, 15, (h, w, 3)).astype(np.uint8)
    elif mode == 'edge':
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mean_color = reference_img.mean(axis=(0, 1))
        canvas[:] = mean_color
        return canvas
    else:
        return np.zeros((h, w, 3), dtype=np.uint8)


def resize_with_padding(img, target_w, target_h, padding_augmentation, padding_mode):
    """Resize image with padding to target size"""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if padding_augmentation:
        canvas = _create_augmented_canvas(target_h, target_w, resized, padding_mode)
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, (scale, x_offset, y_offset)


def transform_keypoints(keypoints, crop_offset, resize_params):
    """Transform keypoints from original frame to crop coordinates"""
    if not keypoints or len(keypoints) < 3:
        return []

    scale, x_offset, y_offset = resize_params
    crop_x, crop_y = crop_offset

    transformed = []
    try:
        for i in range(0, len(keypoints), 3):
            if i + 2 >= len(keypoints):
                break

            x = float(keypoints[i])
            y = float(keypoints[i + 1])
            conf = float(keypoints[i + 2])

            # Transform to crop coordinates
            x_crop = (x - crop_x) * scale + x_offset
            y_crop = (y - crop_y) * scale + y_offset

            transformed.extend([x_crop, y_crop, conf])
    except (TypeError, ValueError, IndexError) as e:
        # If transformation fails, return empty list
        return []

    return transformed


def draw_keypoints(img, keypoints, skeleton=None):
    """Draw keypoints and skeleton on image"""
    img_vis = img.copy()

    if not keypoints or len(keypoints) < 3:
        return img_vis

    # Parse keypoints
    kps = []
    try:
        for i in range(0, len(keypoints), 3):
            if i + 2 >= len(keypoints):
                break
            x = float(keypoints[i])
            y = float(keypoints[i + 1])
            conf = float(keypoints[i + 2])
            kps.append((int(x), int(y), conf))
    except (TypeError, ValueError, IndexError):
        return img_vis

    if not kps:
        return img_vis

    # Auto-generate skeleton connections for non-standard keypoint counts
    # Connect consecutive keypoints in a chain
    num_kps = len(kps)
    if skeleton is None or not skeleton:
        # Simple chain connection for visualization
        skeleton_auto = [[i, i + 1] for i in range(1, num_kps)]
    else:
        skeleton_auto = skeleton

    # Draw skeleton lines
    for connection in skeleton_auto:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        if 0 <= idx1 < len(kps) and 0 <= idx2 < len(kps):
            pt1, pt2 = kps[idx1], kps[idx2]
            if pt1[2] > 0.3 and pt2[2] > 0.3:  # confidence threshold
                cv2.line(img_vis, (pt1[0], pt1[1]), (pt2[0], pt2[1]),
                         (0, 255, 0), 2, cv2.LINE_AA)

    # Draw keypoints on top
    for x, y, conf in kps:
        if conf > 0.3:
            # Outer white circle
            cv2.circle(img_vis, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
            # Inner colored circle
            cv2.circle(img_vis, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    return img_vis


def export_tracklet_clip(cap, tracklet, out_folder, video_fps, original_fps,
                         padding, min_size, padding_augmentation, padding_mode):
    """
    Export tracklet as video (normal + keypoints visualization)
    Returns metadata dict or None
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    start_frame = tracklet[0][0]
    end_frame = tracklet[-1][0]

    # Calculate timestamps
    start_time = start_frame / original_fps
    end_time = end_frame / original_fps

    # Prepare frame lookup
    frame_data = {f: (bbox, kps) for f, bbox, kps in tracklet}

    # Initialize writers
    writer_normal = None
    writer_keypoints = None
    target_w = target_h = None
    fourcc = cv2.VideoWriter_fourcc(*OUT_VIDEO_CODEC)

    metadata = {
        'start_frame': int(start_frame),
        'end_frame': int(end_frame),
        'start_time': float(start_time),
        'end_time': float(end_time),
        'duration': float(end_time - start_time),
        'frames': []
    }

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0

    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if f not in frame_data:
            continue

        bbox, keypoints = frame_data[f]
        crop, crop_offset = extract_crop_from_frame(frame, bbox, padding, min_size)

        if crop is None:
            continue

        # Initialize writers with first valid crop
        if writer_normal is None:
            h0, w0 = crop.shape[:2]
            target_w, target_h = w0, h0

            path_normal = str(out_folder / "clip.mp4")
            path_keypoints = str(out_folder / "clip_keypoints.mp4")

            writer_normal = cv2.VideoWriter(path_normal, fourcc,
                                            float(video_fps), (target_w, target_h))
            writer_keypoints = cv2.VideoWriter(path_keypoints, fourcc,
                                               float(video_fps), (target_w, target_h))

            if not writer_normal.isOpened() or not writer_keypoints.isOpened():
                return None

        # Resize with padding
        crop_padded, resize_params = resize_with_padding(
            crop, target_w, target_h, padding_augmentation, padding_mode)

        # Transform keypoints to crop coordinates
        kps_transformed = transform_keypoints(keypoints, crop_offset, resize_params)

        # Draw keypoints version
        crop_with_kps = draw_keypoints(crop_padded, kps_transformed)

        # Write frames
        writer_normal.write(crop_padded)
        writer_keypoints.write(crop_with_kps)

        # Store metadata
        metadata['frames'].append({
            'frame_idx': int(f),
            'timestamp': float(f / original_fps),
            'bbox': list(map(float, bbox)),
            'keypoints_crop': [float(k) for k in kps_transformed]
        })

        frames_written += 1

    if writer_normal:
        writer_normal.release()
    if writer_keypoints:
        writer_keypoints.release()

    if frames_written == 0:
        return None

    metadata['num_frames'] = frames_written

    # Save metadata
    with open(out_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def process_single_track(args_tuple):
    """Process single track (for multiprocessing)"""
    tid, tdata, video_path, outdir, gap, fps, orig_fps, padding, min_size, pad_aug, pad_mode = args_tuple

    person_folder = outdir / f"person_{tid:04d}"
    person_folder.mkdir(exist_ok=True, parents=True)

    # Group into tracklets
    tracklets = group_into_tracklets(tdata, gap_threshold=gap)

    # Open video for this process
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    clips_exported = 0

    for i, tracklet in enumerate(tracklets, start=1):
        clip_folder = person_folder / f"clip_{i:02d}"

        result = export_tracklet_clip(
            cap=cap,
            tracklet=tracklet,
            out_folder=clip_folder,
            video_fps=fps,
            original_fps=orig_fps,
            padding=padding,
            min_size=min_size,
            padding_augmentation=pad_aug,
            padding_mode=pad_mode
        )

        if result:
            clips_exported += 1
        else:
            # Remove empty folder
            try:
                if clip_folder.exists() and not any(clip_folder.iterdir()):
                    clip_folder.rmdir()
            except:
                pass

    cap.release()
    return clips_exported


def main(args):
    tracks = load_tracks(args.track_json)
    video_path = args.video
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Get original video FPS
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    cap.release()

    print(f" Video: {video_path}")
    print(f" Found {len(tracks)} tracks")
    print(f" Output: {outdir}")
    print(f"  Original FPS: {original_fps:.2f}, Export FPS: {args.fps}")
    print(f" Workers: {args.workers}")

    # Prepare arguments for parallel processing
    process_args = [
        (tid, tdata, video_path, outdir, args.gap, args.fps, original_fps,
         args.padding, args.min_size, not args.no_padding_augmentation, args.padding_mode)
        for tid, tdata in sorted(tracks.items())
    ]

    # Process with multiprocessing
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_track, process_args),
                total=len(process_args),
                desc="Processing tracks"
            ))
    else:
        results = [process_single_track(arg) for arg in tqdm(process_args, desc="Processing tracks")]

    total_clips = sum(results)
    print(f" Export abgeschlossen: {total_clips} clips erstellt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export Tracklets als Videos mit Keypoint-Visualisierung und Metadata"
    )
    p.add_argument("--track-json", required=True, help="Tracker JSON mit keypoints")
    p.add_argument("--video", required=True, help="Original video path")
    p.add_argument("--out-dir", default="track_crops", help="Output base directory")
    p.add_argument("--gap", type=int, default=10,
                   help="Max frame-gap für tracklet merging")
    p.add_argument("--padding", type=int, default=10, help="Crop padding in pixels")
    p.add_argument("--min-size", type=int, default=50, help="Min crop size")
    p.add_argument("--fps", type=int, default=30, help="FPS für export videos")
    p.add_argument("--padding-mode", type=str, default="black",
                   choices=['black', 'blur', 'noise', 'edge', 'random'],
                   help="Padding augmentation mode")
    p.add_argument("--no-padding-augmentation", action="store_true",
                   help="Disable padding augmentation")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers (default: 4)")

    args = p.parse_args()
    main(args)