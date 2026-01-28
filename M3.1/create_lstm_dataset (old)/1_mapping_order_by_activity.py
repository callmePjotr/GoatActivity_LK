import os
import json
from pathlib import Path
import cv2
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional
import shutil

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
BASE_DIR = r"D:\temp_data\sorted\27_08\64256_id3"
BORIS_TSV = r"D:\temp_data\sorted\27_08\64256.tsv"
FPS = 25  # Fixed FPS for all clips
MAX_GAP_FRAMES = 50  # Maximum frame gap to link clips into chains
OUTPUT_BASE = None  # Set to a path if you want different output location, else uses BASE_DIR


# -------------------------------------------------------------
# Load BORIS TSV and convert START/STOP into intervals
# -------------------------------------------------------------
def load_boris_intervals(tsv_path):
    """Load and parse BORIS annotation TSV file."""
    df = pd.read_csv(tsv_path, sep="\t")

    required_cols = ["Subject", "Behavior", "Behavior type", "Time", "Media file name"]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"TSV missing required columns: {required_cols}")

    intervals = []
    grouped = df.groupby(["Subject", "Behavior", "Media file name"])

    for (subject, behavior, mediafile), group in grouped:
        group = group.sort_values("Time").reset_index(drop=True)
        start_time = None

        for _, row in group.iterrows():
            if row["Behavior type"] == "START":
                start_time = row["Time"]
            elif row["Behavior type"] == "STOP":
                if start_time is not None:
                    intervals.append({
                        "subject": subject,
                        "behavior": behavior,
                        "mediafile": mediafile,
                        "start_time": float(start_time),
                        "end_time": float(row["Time"])
                    })
                    start_time = None

        if start_time is not None:
            intervals.append({
                "subject": subject,
                "behavior": behavior,
                "mediafile": mediafile,
                "start_time": float(start_time),
                "end_time": float(start_time)
            })

    return pd.DataFrame(intervals)


# -------------------------------------------------------------
# Build clip chains based on frame continuity
# -------------------------------------------------------------
def build_clip_chains(base_dir: Path, max_gap_frames: int = 50) -> List[List[Dict]]:
    """
    Analyze all clips and build chains based on frame continuity.

    Returns:
        List of chains, where each chain is a list of clip info dicts.
    """
    print("\n📊 Analyzing clip continuity...")

    clips = []

    # Collect all clips with metadata
    for clip_path in base_dir.rglob("clip.mp4"):
        meta_path = clip_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)

                clips.append({
                    'clip_path': clip_path,
                    'meta_path': meta_path,
                    'clip_folder': clip_path.parent,
                    'start_frame': meta.get('start_frame'),
                    'end_frame': meta.get('end_frame'),
                    'metadata': meta
                })

    if not clips:
        print("  ⚠️  No clips found!")
        return []

    # Sort by start_frame
    clips.sort(key=lambda x: x['start_frame'])

    # Build chains
    chains = []
    current_chain = [clips[0]]

    for i in range(1, len(clips)):
        prev_clip = current_chain[-1]
        curr_clip = clips[i]

        gap = curr_clip['start_frame'] - prev_clip['end_frame']

        if gap <= max_gap_frames:
            # Continue chain
            current_chain.append(curr_clip)
        else:
            # Start new chain
            chains.append(current_chain)
            current_chain = [curr_clip]

    # Don't forget last chain
    if current_chain:
        chains.append(current_chain)

    # Print statistics
    print(f"\n  ✓ Found {len(clips)} total clips")
    print(f"  ✓ Organized into {len(chains)} chains")
    print(f"\n  Chain Statistics:")
    for i, chain in enumerate(chains):
        total_frames = sum(c['end_frame'] - c['start_frame'] for c in chain)
        duration_sec = total_frames / FPS
        print(f"    Chain {i}: {len(chain)} clips, {total_frames} frames ({duration_sec:.1f}s)")

    return chains


# -------------------------------------------------------------
# Update metadata with chain information
# -------------------------------------------------------------
def update_metadata_with_chains(chains: List[List[Dict]]) -> None:
    """
    Add chain information to all clip metadata files.
    """
    print("\n🔗 Adding chain information to metadata...")

    for chain_idx, chain in enumerate(chains):
        for pos_in_chain, clip_info in enumerate(chain):
            meta = clip_info['metadata']

            # Add chain information
            meta['chain_id'] = f"chain_{chain_idx}"
            meta['position_in_chain'] = pos_in_chain
            meta['total_clips_in_chain'] = len(chain)

            # Add links to previous/next clips
            if pos_in_chain > 0:
                prev_clip = chain[pos_in_chain - 1]
                meta['previous_clip'] = str(prev_clip['clip_path'].relative_to(BASE_DIR))
                meta['gap_to_previous'] = clip_info['start_frame'] - prev_clip['end_frame']
            else:
                meta['previous_clip'] = None
                meta['gap_to_previous'] = None

            if pos_in_chain < len(chain) - 1:
                next_clip = chain[pos_in_chain + 1]
                meta['next_clip'] = str(next_clip['clip_path'].relative_to(BASE_DIR))
                meta['gap_to_next'] = next_clip['start_frame'] - clip_info['end_frame']
            else:
                meta['next_clip'] = None
                meta['gap_to_next'] = None

            # Save updated metadata
            with open(clip_info['meta_path'], 'w') as f:
                json.dump(meta, f, indent=2)

    print(f"  ✓ Updated {sum(len(chain) for chain in chains)} metadata files")


# -------------------------------------------------------------
# Get activity segments for a clip
# -------------------------------------------------------------
def get_activity_segments(clip_start, clip_end, boris_intervals):
    """
    Returns list of activity segments within clip timerange.
    Each segment: {'behavior': str, 'start': float, 'end': float}
    """
    segments = []

    # Find all intervals that overlap with clip
    overlapping = boris_intervals[
        (boris_intervals["start_time"] < clip_end) &
        (boris_intervals["end_time"] > clip_start)
        ]

    if len(overlapping) == 0:
        # No activity -> create "unknown" segment
        segments.append({
            'behavior': 'unknown',
            'start': clip_start,
            'end': clip_end
        })
        return segments

    # Sort by start time
    overlapping = overlapping.sort_values("start_time").reset_index(drop=True)

    current_time = clip_start

    for _, row in overlapping.iterrows():
        interval_start = max(row["start_time"], clip_start)
        interval_end = min(row["end_time"], clip_end)

        # Gap before this interval
        if current_time < interval_start:
            segments.append({
                'behavior': 'unknown',
                'start': current_time,
                'end': interval_start
            })

        # The actual interval
        segments.append({
            'behavior': row["behavior"],
            'start': interval_start,
            'end': interval_end
        })

        current_time = interval_end

    # Gap after last interval
    if current_time < clip_end:
        segments.append({
            'behavior': 'unknown',
            'start': current_time,
            'end': clip_end
        })

    return segments


# -------------------------------------------------------------
# Split clip by activities (PRESERVING ALL METADATA)
# -------------------------------------------------------------
def split_clip_by_activities(clip_path, metadata_path, boris_intervals, output_base):
    """
    Reads clip and metadata, splits by activity changes,
    saves new clips in activity folders with FULL metadata preservation.
    """
    with open(metadata_path, "r") as f:
        original_meta = json.load(f)

    clip_start = original_meta["start_time"]
    clip_end = original_meta["end_time"]
    clip_start_frame = original_meta["start_frame"]

    # Get activity segments
    segments = get_activity_segments(clip_start, clip_end, boris_intervals)

    if not segments:
        print(f"  ⚠️  No segments found for {clip_path}")
        return 0

    # Open video
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  ❌ Could not open {clip_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Build frame lookup from original metadata
    frame_lookup = {}
    if "frames" in original_meta:
        for frame_info in original_meta["frames"]:
            frame_idx = frame_info["frame_idx"]
            frame_lookup[frame_idx] = frame_info

    clips_created = 0

    # Process each segment
    for seg_idx, segment in enumerate(segments):
        behavior = segment["behavior"]
        seg_start = segment["start"]
        seg_end = segment["end"]

        # Convert times to frame indices
        start_frame_local = int((seg_start - clip_start) * fps)
        end_frame_local = int((seg_end - clip_start) * fps)

        # Calculate global frame indices
        start_frame_global = clip_start_frame + start_frame_local
        end_frame_global = clip_start_frame + end_frame_local

        if end_frame_local <= start_frame_local:
            continue

        # Create activity folder
        activity_folder = Path(output_base) / behavior
        activity_folder.mkdir(exist_ok=True, parents=True)

        # Generate unique clip name based on frame range
        clip_name = f"frames_{start_frame_global:06d}_{end_frame_global:06d}"
        clip_output_path = activity_folder / f"{clip_name}.mp4"
        meta_output_path = activity_folder / f"{clip_name}_metadata.json"

        # Create video writer
        writer = cv2.VideoWriter(str(clip_output_path), fourcc, fps, (width, height))

        # Prepare new metadata - PRESERVE ALL ORIGINAL FIELDS
        new_meta = {
            # Original clip reference
            "original_clip_path": str(clip_path),
            "source_clip_folder": str(clip_path.parent.relative_to(Path(output_base).parent)),

            # Activity information
            "activity": behavior,
            "segment_index": seg_idx,

            # Frame and time information
            "start_frame": int(start_frame_global),
            "end_frame": int(end_frame_global),
            "start_time": float(seg_start),
            "end_time": float(seg_end),
            "duration": float(seg_end - seg_start),

            # Chain information (preserved from original)
            "chain_id": original_meta.get("chain_id"),
            "position_in_chain": original_meta.get("position_in_chain"),
            "total_clips_in_chain": original_meta.get("total_clips_in_chain"),
            "previous_clip": original_meta.get("previous_clip"),
            "next_clip": original_meta.get("next_clip"),
            "gap_to_previous": original_meta.get("gap_to_previous"),
            "gap_to_next": original_meta.get("gap_to_next"),

            # Frame-by-frame data with keypoints
            "frames": []
        }

        # Reset video to segment start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_local)

        frames_written = 0

        # Write frames for this segment - PRESERVE KEYPOINTS
        for local_frame_idx in range(start_frame_local, end_frame_local):
            ret, frame = cap.read()
            if not ret:
                break

            writer.write(frame)

            # Get global frame index
            global_frame_idx = clip_start_frame + local_frame_idx

            # Copy complete frame metadata including keypoints
            if global_frame_idx in frame_lookup:
                # IMPORTANT: Deep copy to preserve all fields
                frame_info = frame_lookup[global_frame_idx].copy()
                new_meta["frames"].append(frame_info)
            else:
                # Fallback: create minimal frame info (shouldn't happen normally)
                timestamp = seg_start + (frames_written / fps)
                new_meta["frames"].append({
                    "frame_idx": int(global_frame_idx),
                    "timestamp": float(timestamp),
                    "bbox": [0, 0, 0, 0],
                    "keypoints_crop": []
                })

            frames_written += 1

        writer.release()

        if frames_written > 0:
            new_meta["num_frames"] = frames_written

            # Save metadata with preserved keypoints
            with open(meta_output_path, "w") as f:
                json.dump(new_meta, f, indent=2)

            # Verify keypoints were preserved
            keypoint_frames = sum(1 for f in new_meta["frames"] if f.get("keypoints_crop"))
            print(f"  ✓ Created: {clip_output_path.name}")
            print(f"    Frames: {frames_written}, Activity: {behavior}, Keypoints: {keypoint_frames}/{frames_written}")
            clips_created += 1
        else:
            # Remove empty video
            if clip_output_path.exists():
                clip_output_path.unlink()

    cap.release()
    return clips_created


# -------------------------------------------------------------
# Generate summary statistics
# -------------------------------------------------------------
def generate_summary(output_base: Path):
    """Generate summary statistics about the processed clips."""
    print("\n" + "=" * 70)
    print("📈 SUMMARY STATISTICS")
    print("=" * 70)

    activity_stats = defaultdict(lambda: {"clips": 0, "frames": 0, "duration": 0.0})

    for activity_folder in output_base.iterdir():
        if activity_folder.is_dir():
            activity = activity_folder.name

            for meta_file in activity_folder.glob("*_metadata.json"):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    activity_stats[activity]["clips"] += 1
                    activity_stats[activity]["frames"] += meta.get("num_frames", 0)
                    activity_stats[activity]["duration"] += meta.get("duration", 0.0)

    print(f"\n{'Activity':<20} {'Clips':<10} {'Frames':<10} {'Duration (s)':<15}")
    print("-" * 70)

    total_clips = 0
    total_frames = 0
    total_duration = 0.0

    for activity in sorted(activity_stats.keys()):
        stats = activity_stats[activity]
        print(f"{activity:<20} {stats['clips']:<10} {stats['frames']:<10} {stats['duration']:<15.2f}")
        total_clips += stats['clips']
        total_frames += stats['frames']
        total_duration += stats['duration']

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_clips:<10} {total_frames:<10} {total_duration:<15.2f}")
    print("=" * 70)


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("🎬 ACTIVITY MAPPING WITH CLIP CHAINS")
    print("=" * 70)

    # Set output directory
    output_base = Path(OUTPUT_BASE) if OUTPUT_BASE else Path(BASE_DIR)
    base_path = Path(BASE_DIR)

    # Step 1: Build clip chains
    chains = build_clip_chains(base_path, MAX_GAP_FRAMES)

    if not chains:
        print("\n❌ No clips found to process!")
        return

    # Step 2: Update metadata with chain information
    update_metadata_with_chains(chains)

    # Step 3: Load BORIS annotations
    print("\n📋 Loading BORIS annotations...")
    boris_intervals = load_boris_intervals(BORIS_TSV)
    print(f"  ✓ Loaded {len(boris_intervals)} behavior intervals")

    # Step 4: Process all clips
    print("\n🎥 Processing clips and splitting by activity...")
    total_clips = 0
    processed_clips = 0

    for clip_info in [clip for chain in chains for clip in chain]:
        clip_path = clip_info['clip_path']
        meta_path = clip_info['meta_path']

        print(f"\n▶ Processing: {clip_path.relative_to(base_path)}")
        clips_created = split_clip_by_activities(
            clip_path, meta_path, boris_intervals, output_base
        )
        total_clips += clips_created
        processed_clips += 1

    # Step 5: Generate summary
    generate_summary(output_base)

    print(f"\n{'=' * 70}")
    print(f"🎉 FINISHED!")
    print(f"   Processed: {processed_clips} original clips")
    print(f"   Created: {total_clips} activity-labeled clips")
    print(f"   Output directory: {output_base}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()