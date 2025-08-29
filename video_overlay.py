# Last modified: 2025-08-29 11:05:43
appVersion = "0.1.11"
import cv2
import csv
import os

VIDEO_PATH = "/ai/bennwittRepos/velocityView/output/recorded_video.mp4"
OUTPUT_PATH = "/ai/bennwittRepos/velocityView/output/velocity_overlay.mp4"
FPS_FALLBACK = 24.0
RESOLUTION = None  # Auto from input unless overridden
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

# Determine FPS and resolution dynamically
in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
in_fps = cap.get(cv2.CAP_PROP_FPS)
fps = in_fps if in_fps and in_fps > 1.0 else FPS_FALLBACK
out_size = RESOLUTION if RESOLUTION else (in_w, in_h)

# MP4 writer using mp4v codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, out_size)
if not out.isOpened():
    cap.release()
    raise RuntimeError(
        "❌ Failed to open MP4 writer. Your OpenCV may lack MP4 support."
    )

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

    # Resize if desired output size differs
    if (frame.shape[1], frame.shape[0]) != out_size:
        frame = cv2.resize(frame, out_size)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Annotated video saved to: {OUTPUT_PATH}")
