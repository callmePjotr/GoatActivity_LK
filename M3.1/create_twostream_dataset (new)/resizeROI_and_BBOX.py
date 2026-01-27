import os
import json
import cv2
from collections import defaultdict
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================
ROOT_DIR = "output_dir_der_sortierten_Ziegen" # wichtig aber optional !!! --> alle sortierten Tiere in einen Ordner
VIDEO_PATH = "pfad_zum_originalvideo/video.mp4"
SUBCLIP_ROOT = "output_dir"

BBOX_SIZE = 224
ROI_SIZE = 224
ROI_PADDING = 100

# ==========================================================
# RESIZE STRATEGY
# ==========================================================
RESIZE_STRATEGY = "stretch"  # "stretch", "crop_center", "adaptive_padding"
PADDING_COLOR = (114, 114, 114)

# ==========================================================
# RESIZE FUNCTIONS
# ==========================================================
def resize_stretch(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def resize_crop_center(img, w, h):
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    x0 = (nw - w) // 2
    y0 = (nh - h) // 2
    return resized[y0:y0+h, x0:x0+w]

def resize_adaptive_padding(img, w, h, color):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

def apply_resize(img, size, strategy):
    if strategy == "stretch":
        return resize_stretch(img, size, size)
    if strategy == "crop_center":
        return resize_crop_center(img, size, size)
    if strategy == "adaptive_padding":
        return resize_adaptive_padding(img, size, size, PADDING_COLOR)
    raise ValueError(strategy)

# ==========================================================
# LOAD ORIGINAL METADATA
# ==========================================================
frame_index_map = defaultdict(list)
original_metadata = {}

for root, _, files in os.walk(ROOT_DIR):
    if "metadata.json" not in files:
        continue

    clip = os.path.basename(root)
    tier = os.path.basename(os.path.dirname(root))
    label = f"{tier}__{clip}"

    with open(os.path.join(root, "metadata.json")) as f:
        meta = json.load(f)

    original_metadata[label] = meta

    for fr in meta.get("frames", []):
        frame_index_map[fr["frame_idx"]].append({
            "tier": tier,
            "clip": clip,
            "bbox": fr["bbox"]
        })

if not frame_index_map:
    raise RuntimeError("Keine Annotationen gefunden")

# ==========================================================
# VIDEO IO
# ==========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video konnte nicht geöffnet werden")

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(SUBCLIP_ROOT, exist_ok=True)

# ==========================================================
# SUBCLIP STATE (NO FRAGMENTATION)
# ==========================================================
writers = {}
clip_metadata = {}

def create_writers(label):
    tier, clip = label.split("__")
    out_dir = os.path.join(SUBCLIP_ROOT, tier, clip)
    os.makedirs(out_dir, exist_ok=True)

    wb = cv2.VideoWriter(
        os.path.join(out_dir, "bbox.mp4"),
        fourcc, fps, (BBOX_SIZE, BBOX_SIZE)
    )
    wr = cv2.VideoWriter(
        os.path.join(out_dir, "roi.mp4"),
        fourcc, fps, (ROI_SIZE, ROI_SIZE)
    )
    return wb, wr, out_dir

# ==========================================================
# PROCESS VIDEO
# ==========================================================
frame_idx = 0

with tqdm(total=N) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_index_map:
            for obj in frame_index_map[frame_idx]:
                label = f"{obj['tier']}__{obj['clip']}"

                if label not in writers:
                    wb, wr, out_dir = create_writers(label)
                    writers[label] = (wb, wr, out_dir)

                    meta = original_metadata[label]
                    clip_metadata[label] = {
                        "start_frame": meta["start_frame"],
                        "end_frame": meta["end_frame"],
                        "start_time": meta["start_time"],
                        "end_time": meta["end_time"],
                        "fps": fps,
                        "frames": []
                    }

                x1, y1, x2, y2 = map(int, obj["bbox"])
                x1b, y1b = max(0, x1), max(0, y1)
                x2b, y2b = min(W, x2), min(H, y2)

                bbox_crop = frame[y1b:y2b, x1b:x2b]

                if bbox_crop.size == 0:
                    clip_metadata[label]["frames"].append({
                        "frame_idx": frame_idx,
                        "status": "missing_crop"
                    })
                    continue

                bbox_resized = apply_resize(bbox_crop, BBOX_SIZE, RESIZE_STRATEGY)

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rx1 = max(0, int(cx - ROI_PADDING))
                ry1 = max(0, int(cy - ROI_PADDING))
                rx2 = min(W, int(cx + ROI_PADDING))
                ry2 = min(H, int(cy + ROI_PADDING))

                roi_crop = frame[ry1:ry2, rx1:rx2]
                if roi_crop.size == 0:
                    clip_metadata[label]["frames"].append({
                        "frame_idx": frame_idx,
                        "bbox": [x1, y1, x2, y2],
                        "roi": None,
                        "status": "missing_roi"
                    })
                    continue

                roi_resized = apply_resize(roi_crop, ROI_SIZE, RESIZE_STRATEGY)

                wb, wr, _ = writers[label]
                wb.write(bbox_resized)
                wr.write(roi_resized)

                clip_metadata[label]["frames"].append({
                    "frame_idx": frame_idx,
                    "bbox": [x1, y1, x2, y2],
                    "roi": [rx1, ry1, rx2, ry2],
                    "status": "ok"
                })

        frame_idx += 1
        pbar.update(1)

# ==========================================================
# FINAL SAVE (ONE CLIP PER ORIGINAL)
# ==========================================================
for label, (wb, wr, out_dir) in writers.items():
    wb.release()
    wr.release()

    meta = clip_metadata[label]
    meta["num_frames"] = len(meta["frames"])

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

cap.release()
print("✅ Fertig – exakt EIN Subclip pro Originalclip, keine Fragmentierung")
