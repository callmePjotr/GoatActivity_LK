import os
import json
from pathlib import Path
import cv2
import pandas as pd
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path("sorted_animal_folder/Ziege_1234") # wichtig !!! --> hier nur den einzelnen Ordner EINER Ziege angeben
BORIS_TSV = Path("sorted_animal_Activity/64256.tsv") # die zugehörige Boris Datei
FPS_FALLBACK = 25

# ============================================================
# LOAD BORIS INTERVALS
# ============================================================
def load_boris_intervals(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")

    intervals = []
    grouped = df.groupby(["Subject", "Behavior", "Media file name"])

    for (_, behavior, _), group in grouped:
        group = group.sort_values("Time")
        start = None

        for _, row in group.iterrows():
            if row["Behavior type"] == "START":
                start = row["Time"]
            elif row["Behavior type"] == "STOP" and start is not None:
                intervals.append({
                    "behavior": behavior,
                    "start_time": float(start),
                    "end_time": float(row["Time"])
                })
                start = None

    return pd.DataFrame(intervals)

# ============================================================
# ACTIVITY SEGMENTS
# ============================================================
def get_activity_segments(start, end, intervals):
    segments = []

    overlaps = intervals[
        (intervals.start_time < end) &
        (intervals.end_time > start)
    ].sort_values("start_time")

    if overlaps.empty:
        return [{
            "behavior": "unknown",
            "start": start,
            "end": end
        }]

    t = start
    for _, row in overlaps.iterrows():
        s = max(row.start_time, start)
        e = min(row.end_time, end)

        if t < s:
            segments.append({"behavior": "unknown", "start": t, "end": s})

        segments.append({"behavior": row.behavior, "start": s, "end": e})
        t = e

    if t < end:
        segments.append({"behavior": "unknown", "start": t, "end": end})

    return segments

# ============================================================
# LOAD CLIPS (ROI + BBOX + METADATA REQUIRED)
# ============================================================
def load_clips(base_dir: Path):
    clips = []

    for meta_path in base_dir.rglob("metadata.json"):
        clip_dir = meta_path.parent
        roi = clip_dir / "roi.mp4"
        bbox = clip_dir / "bbox.mp4"

        if not roi.exists() or not bbox.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        clips.append({
            "dir": clip_dir,
            "roi": roi,
            "bbox": bbox,
            "meta": meta
        })

    if not clips:
        raise RuntimeError("❌ No clips found – check directory structure")

    print(f"✓ Found {len(clips)} clips")
    return clips

# ============================================================
# SPLIT ONE CLIP (DUAL STREAM)
# ============================================================
def split_clip(clip, intervals):
    meta = clip["meta"]

    start_t = meta["start_time"]
    end_t = meta["end_time"]
    start_frame = meta["start_frame"]

    segments = get_activity_segments(start_t, end_t, intervals)

    cap_roi = cv2.VideoCapture(str(clip["roi"]))
    cap_bbox = cv2.VideoCapture(str(clip["bbox"]))

    fps = cap_roi.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    w = int(cap_roi.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_roi.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_lookup = {f["frame_idx"]: f for f in meta.get("frames", [])}
    created = 0

    for idx, seg in enumerate(segments):
        l_start = int((seg["start"] - start_t) * fps)
        l_end = int((seg["end"] - start_t) * fps)

        if l_end <= l_start:
            continue

        g_start = start_frame + l_start
        g_end = start_frame + l_end

        out_dir = clip["dir"].parent / seg["behavior"] / f"frames_{g_start:06d}_{g_end:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        roi_out = cv2.VideoWriter(
            str(out_dir / "roi.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )
        bbox_out = cv2.VideoWriter(
            str(out_dir / "bbox.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        cap_roi.set(cv2.CAP_PROP_POS_FRAMES, l_start)
        cap_bbox.set(cv2.CAP_PROP_POS_FRAMES, l_start)

        new_meta = {
            **meta,
            "activity": seg["behavior"],
            "segment_index": idx,
            "start_frame": g_start,
            "end_frame": g_end,
            "start_time": seg["start"],
            "end_time": seg["end"],
            "duration": seg["end"] - seg["start"],
            "frames": []
        }

        for lf in range(l_start, l_end):
            ok1, fr1 = cap_roi.read()
            ok2, fr2 = cap_bbox.read()
            if not ok1 or not ok2:
                break

            roi_out.write(fr1)
            bbox_out.write(fr2)

            gf = start_frame + lf
            if gf in frame_lookup:
                new_meta["frames"].append(frame_lookup[gf])

        roi_out.release()
        bbox_out.release()

        if new_meta["frames"]:
            new_meta["num_frames"] = len(new_meta["frames"])
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            created += 1

    cap_roi.release()
    cap_bbox.release()
    return created

# ============================================================
# MAIN
# ============================================================
def main():
    print("🎬 Two-Stream Activity Splitter")

    intervals = load_boris_intervals(BORIS_TSV)
    clips = load_clips(BASE_DIR)

    total = 0
    for clip in clips:
        print(f"▶ {clip['dir'].name}")
        total += split_clip(clip, intervals)

    print(f"\n✅ Done – created {total} activity clips")

if __name__ == "__main__":
    main()
