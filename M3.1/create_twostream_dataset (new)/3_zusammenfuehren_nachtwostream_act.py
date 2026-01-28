import json
import shutil
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SOURCE_BASE = Path("pfad_zu_den_sortierten_gemappten_tieren") # wichtig !!! --> hier wieder den Ordner mit allen Tieren angeben, in den Clip Ordnern entstehen nun die Aktivitätsordner
OUTPUT_BASE = Path("output_folder")

ACTIVITIES = {"Moving", "Standing", "Lying", "unknown", "Climbing", "Perturbation", "Rearing"} # hier später andere Logik anwenden

# ============================================================
# MERGE DATASET
# ============================================================
def merge_all_animals(source_base: Path, output_base: Path):
    output_base.mkdir(parents=True, exist_ok=True)

    total_clips = 0

    for animal_dir in source_base.iterdir():
        if not animal_dir.is_dir():
            continue

        animal_id = animal_dir.name
        print(f"\n🐄 Processing animal: {animal_id}")

        for activity_dir in animal_dir.iterdir():
            if not activity_dir.is_dir():
                continue

            activity = activity_dir.name
            if activity not in ACTIVITIES:
                continue

            target_activity_dir = output_base / activity
            target_activity_dir.mkdir(exist_ok=True)

            for frame_dir in activity_dir.iterdir():
                if not frame_dir.is_dir():
                    continue

                roi = frame_dir / "roi.mp4"
                bbox = frame_dir / "bbox.mp4"
                meta = frame_dir / "metadata.json"

                if not roi.exists() or not bbox.exists() or not meta.exists():
                    print(f"  ⚠️ Skipped incomplete clip: {frame_dir}")
                    continue

                new_clip_name = f"{animal_id}_{frame_dir.name}"
                target_dir = target_activity_dir / new_clip_name

                if target_dir.exists():
                    print(f"  ⚠️ Already exists, skipping: {target_dir}")
                    continue

                target_dir.mkdir(parents=True)

                # Copy videos
                shutil.copy2(roi, target_dir / "roi.mp4")
                shutil.copy2(bbox, target_dir / "bbox.mp4")

                # Load & extend metadata
                with open(meta, "r") as f:
                    metadata = json.load(f)

                metadata["animal_id"] = animal_id
                metadata["activity"] = activity
                metadata["source_path"] = str(frame_dir)

                with open(target_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                total_clips += 1

    print("\n" + "=" * 60)
    print(f"✅ MERGE COMPLETE")
    print(f"   Total clips: {total_clips}")
    print(f"   Output: {output_base}")
    print("=" * 60)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    merge_all_animals(SOURCE_BASE, OUTPUT_BASE)
