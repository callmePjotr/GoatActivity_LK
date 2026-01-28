import cv2
import json
from pathlib import Path
import math

# =========================
# CONFIG
# =========================
INPUT_ROOT = Path(r"D:\temp_data\sorted\normalized_clips_full")
OUTPUT_ROOT = Path(r"D:\temp_data\sorted\windowed_dataset_8fps_64win_32stride")

TARGET_FPS = 8
WINDOW_SIZE = 64
STRIDE = 32

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# CORE FUNCTIONS
# =========================

def process_clip(video_path: Path, json_path: Path, out_activity_dir: Path):
    with open(json_path, "r") as f:
        meta = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    if orig_fps <= 0:
        print(f"[WARN] FPS konnte nicht gelesen werden: {video_path}")
        cap.release()
        return

    step = max(1, int(round(orig_fps / TARGET_FPS)))
    actual_fps = orig_fps / step

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_meta = meta["frames"]
    frame_lookup = {f["frame_idx"]: f for f in frames_meta}

    down_frames = []
    down_images = []

    frame_counter = 0
    kept_idx = 0
    start_frame = meta["start_frame"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_idx = start_frame + frame_counter

        if frame_counter % step == 0 and orig_idx in frame_lookup:
            down_images.append(frame)

            fm = frame_lookup[orig_idx].copy()
            fm["frame_idx"] = kept_idx
            fm["timestamp"] = kept_idx / actual_fps
            down_frames.append(fm)

            kept_idx += 1

        frame_counter += 1

    cap.release()

    if len(down_images) < WINDOW_SIZE:
        return  # zu kurz für ein Window

    # =========================
    # SLIDING WINDOWS
    # =========================

    num_windows = math.floor((len(down_images) - WINDOW_SIZE) / STRIDE) + 1

    for w in range(num_windows):
        start = w * STRIDE
        end = start + WINDOW_SIZE

        win_video = out_activity_dir / f"{video_path.stem}_win_{w:03d}.mp4"
        win_json = out_activity_dir / f"{video_path.stem}_win_{w:03d}.json"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(win_video), fourcc, actual_fps, (width, height))

        for img in down_images[start:end]:
            writer.write(img)

        writer.release()

        win_meta = meta.copy()
        win_meta["frames"] = down_frames[start:end]
        win_meta["fps"] = actual_fps
        win_meta["original_fps"] = orig_fps
        win_meta["num_frames"] = WINDOW_SIZE
        win_meta["start_frame"] = start
        win_meta["end_frame"] = end - 1
        win_meta["start_time"] = start / actual_fps
        win_meta["end_time"] = end / actual_fps
        win_meta["duration"] = WINDOW_SIZE / actual_fps
        win_meta["window_index"] = w
        win_meta["source_clip"] = video_path.name

        with open(win_json, "w") as f:
            json.dump(win_meta, f, indent=2)

    print(f"[OK] {video_path.name} → {num_windows} Windows")


# =========================
# MAIN LOOP
# =========================

def run():
    for activity_dir in INPUT_ROOT.iterdir():
        if not activity_dir.is_dir():
            continue

        out_activity_dir = OUTPUT_ROOT / activity_dir.name
        out_activity_dir.mkdir(parents=True, exist_ok=True)

        for video_path in activity_dir.glob("*.mp4"):
            json_path = video_path.with_name(video_path.stem + "_metadata.json")
            if not json_path.exists():
                print(f"[WARN] Keine JSON für {video_path.name}")
                continue

            process_clip(video_path, json_path, out_activity_dir)


if __name__ == "__main__":
    run()
