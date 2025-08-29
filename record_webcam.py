# Last modified: 2025-08-28 13:40:37
appVersion = "0.0.42"
import cv2
import time
import os
import csv
import threading

# Config
device_index = 0  # Usually /dev/video0
output_file = "/ai/bennwittRepos/velocityView/output/recorded_video.mp4"
fps = 24  # Target 24 FPS output
duration_sec = 10
# We will sample frames on a fixed 24Hz schedule using the latest captured frame
resolution = (1280, 720)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Setup capture
cap = cv2.VideoCapture(device_index)
# Many webcams (e.g., Logitech C922) need MJPG to hit 720p@60fps
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
try:
    # Reduce internal buffering to minimize latency/jitter if backend supports it
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

if not cap.isOpened():
    raise RuntimeError(
        "❌ Failed to open webcam (device index {})".format(device_index)
    )

# Query actual capture settings and setup writer (MP4 container)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (actual_w, actual_h))

if not out.isOpened():
    cap.release()
    raise RuntimeError(
        "❌ Failed to open video writer for MP4. Your OpenCV build may lack MP4 support."
    )

print(
    f"🎥 Recording ~{duration_sec:.2f}s at {fps} FPS (scheduled). "
    f"Camera reports {actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
)

frame_count = 0


# Background frame grabber to timestamp frames as they arrive
class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = None  # perf_counter timestamp when frame was read
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            ts = time.perf_counter()
            if not ret:
                # Short pause to avoid hot loop if camera hiccups
                time.sleep(0.002)
                continue
            with self.lock:
                self.latest_frame = frame
                self.latest_ts = ts

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None
            # Return a copy to avoid mutation issues
            return self.latest_frame.copy(), self.latest_ts

    def stop(self):
        self.running = False


grabber = FrameGrabber(cap)
grabber.start()


def format_timecode(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2
color_red = (0, 0, 255)  # BGR
margin = 10

# Wait for first captured frame to anchor start time
first_frame, first_ts = None, None
print("⌛ Waiting for first frame...")
while True:
    frame, ts = grabber.get_latest()
    if frame is not None:
        first_frame, first_ts = frame, ts
        break
    time.sleep(0.001)

# Initialize timeline to the moment the first frame arrived
start_time = first_ts
frame_interval = 1.0 / fps
target_frames = int(round(duration_sec * fps))
next_frame_time = start_time

# Prepare CSV log for precise timestamps
csv_path = os.path.splitext(output_file)[0] + ".csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["frame_index", "capture_perf", "elapsed_sec", "write_perf"]
)  # headers

print(f"🚀 Starting 24 FPS schedule at t0={start_time:.6f}")

while frame_count < target_frames:
    now = time.perf_counter()
    delay = next_frame_time - now
    if delay > 0:
        # Minimal sleep to reduce CPU while holding tight schedule
        time.sleep(min(0.0005, delay))
        continue

    # Get the most recent frame at this tick
    frame, ts = grabber.get_latest()
    if frame is None or ts is None:
        # No frame yet; advance schedule but don't write to keep timing true
        next_frame_time += frame_interval
        continue

    # Ensure frame size matches writer
    if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
        frame = cv2.resize(frame, (actual_w, actual_h))

    # Real elapsed time based on capture timestamp of this frame
    elapsed = max(0.0, ts - start_time)
    text = format_timecode(elapsed)

    # Position centered horizontally, at top of the frame
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (actual_w - text_w) // 2)
    y = margin + text_h  # top edge, account for baseline
    cv2.putText(
        frame, text, (x, y), font, font_scale, color_red, thickness, cv2.LINE_AA
    )

    out.write(frame)
    frame_count += 1
    write_ts = time.perf_counter()
    csv_writer.writerow(
        [frame_count - 1, f"{ts:.9f}", f"{elapsed:.9f}", f"{write_ts:.9f}"]
    )
    next_frame_time += frame_interval

    if frame_count % int(max(1, fps)) == 0:
        print(f"  ⏱️  {frame_count} frames written (elapsed {elapsed:.3f}s)")

# Cleanup
grabber.stop()
cap.release()
out.release()
csv_file.flush()
csv_file.close()
total_elapsed = time.perf_counter() - start_time
achieved_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
print(f"✅ Saved {frame_count} frames to: {output_file}")
print(f"🧾 Per-frame timestamps: {csv_path}")
print(f"📈 Achieved ~{achieved_fps:.2f} FPS over {total_elapsed:.3f}s")
