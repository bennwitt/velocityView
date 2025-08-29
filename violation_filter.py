# Last modified: 2025-08-29 08:22:37
appVersion = "0.0.3"
import csv
import os

# Config
INPUT_CSV = "output/speed_events.csv"
OUTPUT_CSV = "output/violations.csv"
SPEED_LIMIT_MPH = 25.0  # Set your threshold

print(f"📖 Reading speed events from: {INPUT_CSV}")
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File not found: {INPUT_CSV}")

violations = []

with open(INPUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        speed = float(row["speed_mph"])
        if speed > SPEED_LIMIT_MPH:
            violations.append(row)

print(f"🚨 Found {len(violations)} speed violations over {SPEED_LIMIT_MPH} mph")

# Output
os.makedirs("output", exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["object_id", "start_tick", "end_tick", "frame_delta", "speed_mph"],
    )
    writer.writeheader()
    writer.writerows(violations)

print(f"✅ Violations written to {OUTPUT_CSV}")
