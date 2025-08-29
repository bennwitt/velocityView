# Last modified: 2025-08-29 08:21:50
appVersion = "0.0.4"

import csv
from collections import defaultdict
import os

# Config
INPUT_CSV = "output/detections_log.csv"
OUTPUT_CSV = "output/speed_events.csv"
FPS = 24.0
TICK_SPACING_FT = 10
MIN_TICKS = 2  # Require at least 2 tick crossings to compute speed

# Track: {object_key: [(tick_id, frame)]}
track_data = defaultdict(list)

print("📖 Reading detections...")

with open(INPUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (
            row["class_id"],
            row["x"],
            row["y"],
            row["w"],
            row["h"],
        )  # crude object ID
        frame = int(row["frame"])
        tick_id = int(row["tick_id"])
        if (tick_id, frame) not in track_data[key]:  # Avoid duplicates
            track_data[key].append((tick_id, frame))

print(f"🧠 Processing {len(track_data)} tracked vehicle paths...")

os.makedirs("output", exist_ok=True)
with open(OUTPUT_CSV, "w") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["object_id", "start_tick", "end_tick", "frame_delta", "speed_mph"])

    for i, (obj_key, events) in enumerate(track_data.items()):
        events = sorted(events, key=lambda x: x[1])  # sort by frame
        if len(events) >= MIN_TICKS:
            start_tick, start_frame = events[0]
            end_tick, end_frame = events[1]
            tick_distance = abs(end_tick - start_tick) * TICK_SPACING_FT
            frame_delta = abs(end_frame - start_frame)
            if frame_delta == 0:
                continue
            time_sec = frame_delta / FPS
            speed_mph = (tick_distance / time_sec) * 0.681818
            writer.writerow([i, start_tick, end_tick, frame_delta, round(speed_mph, 2)])

print(f"✅ Speed events written to {OUTPUT_CSV}")
