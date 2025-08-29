# Last modified: 2025-08-29 08:23:11
appVersion = "0.0.5"
import cv2
import csv
import os

VIDEO_PATH = "output/recorded_video.avi"
OUTPUT_PATH = "output/velocity_overlay.avi"
FPS = 24.0
RESOLUTION = (640, 480)
SPEED_EVENTS_FILE = "output/speed_events.csv"
VIOLATIONS_FILE = "output/violations.csv"

# Load violations
violators = set()
with open(VIOLATIONS_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        violators.add(int(row["object_id"]))

# Load speed events
events = {}
with open(SPEED_EVENTS_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        obj_id = int(row["object_id"])
        events[obj_id] = row

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open input video.")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, RESOLUTION)

frame_idx = 0
print("🎥 Annotating video with speed overlays...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for obj_id, row in events.items():
        start_frame = int(row["frame_delta"])  # crude placeholder
        if frame_idx == start_frame:
            # Reconstruct position from detection key
            x, y, w, h = (
                map(int, (row["x"], row["y"], row["w"], row["h"]))
                if all(k in row for k in ["x", "y", "w", "h"])
                else (100, 100, 50, 50)
            )
            speed = float(row["speed_mph"])
            color = (0, 0, 255) if obj_id in violators else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{speed:.1f} mph"
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Annotated video saved to: {OUTPUT_PATH}")
