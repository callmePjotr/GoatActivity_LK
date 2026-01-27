import cv2
import json
import os
import random

# === Parameter anpassen ===
VIDEO_PATH = "path_to_video/video.mp4"
TRACKED_JSON = "path_to_track_json/tracked_output.json"
OUTPUT_VIDEO = "output_video/output.avi"




# === Farben zufällig pro ID generieren ===
def get_color(track_id):
    """Generiert konsistente Farbe für eine Track-ID"""
    random.seed(track_id)
    return tuple(int(x) for x in random.choices(range(50, 255), k=3))


# === JSON laden ===
print("📂 Lade Tracking-Daten...")
with open(TRACKED_JSON, "r") as f:
    data = json.load(f)

# Debug: Struktur prüfen
print(f"JSON-Typ: {type(data)}")
if isinstance(data, dict):
    print(f"Anzahl Frames im JSON: {len(data)}")
    first_key = list(data.keys())[0] if data else None
    if first_key:
        print(f"Beispiel-Frame {first_key}: {len(data[first_key])} Tracks")
        if data[first_key]:
            print(f"Track-Struktur: {data[first_key][0].keys()}")

# === Video öffnen ===
print("\n🎥 Öffne Video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Video nicht gefunden: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✅ Video geladen: {width}x{height}, {fps:.2f} FPS, {total_frames} Frames")

# === Output Video Writer ===
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError(f"❌ Konnte Output-Video nicht erstellen: {OUTPUT_VIDEO}")

frame_idx = 0
font = cv2.FONT_HERSHEY_SIMPLEX
frames_with_tracks = 0
total_tracks_drawn = 0

print("\n🎬 Rendering video with tracks...")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame-Daten aus JSON holen
    # Das Tracking-JSON hat Struktur: {frame_idx: [list of tracks]}
    frame_key = str(frame_idx)
    tracks = []

    if isinstance(data, dict) and frame_key in data:
        tracks = data[frame_key]

    # Tracks zeichnen
    if tracks and len(tracks) > 0:
        frames_with_tracks += 1

        for track in tracks:
            track_id = track.get("track_id")
            bbox = track.get("bbox")
            keypoints = track.get("keypoints", [])

            if track_id is not None and bbox:
                color = get_color(track_id)
                x1, y1, x2, y2 = map(int, bbox)

                # Bounding Box zeichnen
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # ID-Label mit Hintergrund
                label = f"ID {track_id}"
                (label_w, label_h), baseline = cv2.getTextSize(label, font, 0.6, 1)
                # etwas transparenter/heller Hintergrund
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(frame, label, (x1, y1 - 4),
                            font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Keypoints zeichnen
                for kp in keypoints:
                    if len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        # Confidence (falls vorhanden)
                        conf = kp[2] if len(kp) > 2 else 1.0

                        # Nur zeichnen wenn Confidence hoch genug
                        if conf > 0.3:
                            cv2.circle(frame, (x, y), 2, color, -1)  # kleiner Punkt
                            cv2.circle(frame, (x, y), 3, (255, 255, 255), 1)  # dünner Ring

                total_tracks_drawn += 1

    # Frame-Info einblenden
    info_text = f"Frame: {frame_idx} | Tracks: {len(tracks)}"
    cv2.putText(frame, info_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    frame_idx += 1

    # Fortschritt anzeigen (alle 100 Frames)
    if frame_idx % 100 == 0:
        print(f"Verarbeitet: {frame_idx}/{total_frames} Frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✅ Tracking-Video erfolgreich erstellt!")
print("=" * 60)
print(f"📊 Statistik:")
print(f"   Gesamt Frames: {frame_idx}")
print(f"   Frames mit Tracks: {frames_with_tracks}")
print(f"   Gesamt gezeichnete Tracks: {total_tracks_drawn}")
print(f"   Durchschnittliche Tracks/Frame: {total_tracks_drawn / frame_idx:.2f}")
print(f"\n💾 Gespeichert unter:\n   {OUTPUT_VIDEO}")
print("=" * 60)