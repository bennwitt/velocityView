# Last modified: 2025-08-29 17:10:27
appVersion = "0.3.13"
# velocity_infer.py - Phase 1
import cv2
import numpy as np
import time
from datetime import datetime
import os

# Config
MODEL_PATH = "/ai/bennwittRepos/velocityView/models/yolo11n.onnx"
VIDEO_INPUT = 0  # Use 0 for webcam, or path to video
OUTPUT_VIDEO_PATH = "/ai/bennwittRepos/velocityView/output/detections_annotated.mp4"  # legacy single-file path (unused for rolling clips)
FPS_FALLBACK = 24.0
CONFIDENCE_THRESHOLD = 0.54
NMS_THRESHOLD = 0.4
# Number of frames to record starting from the FIRST detection.
# Recording will stop exactly after this many frames, regardless of
# additional detections that may follow while recording is active.
TAIL_FRAMES_AFTER_DETECTION = 300

# COCO class names

COCO_NAMES = [
    "person",  # 0
    "bicycle",  # 1
    "car",  # 2
    "motorcycle",  # 3
    "airplane",
    "bus",
    "train",
    "truck",  # 7
    "boat",  # 8
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",  # 14
    "cat",  # 15
    "dog",  # 16
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",  # 29
    "skis",
    "snowboard",
    "sports ball",  # 32
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",  # 36
    "surfboard",
    "tennis racket",
    "bottle",  # 39
    "wine glass",  # 40
    "cup",
    "fork",
    "knife",  # 43
    "spoon",
    "bowl",
    "banana",  # 46
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# COCO class IDs to detect in residential context
# 0: person, ....
ALLOWED_CLASS_IDS = [0, 1, 2, 3, 7, 8, 14, 15, 16, 29, 32, 36, 39, 40, 43]

# Load network
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
except Exception:
    # Fallback to default backend if CUDA not available
    pass

# Video setup
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise RuntimeError("Failed to open video input")

# Output dir
os.makedirs("output", exist_ok=True)
log_path = "/ai/bennwittRepos/velocityView/output/detections_log.csv"
# Append to log, write header if file is empty
need_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
log_file = open(log_path, "a", buffering=1)
if need_header:
    log_file.write("frame,class_id,class_name,confidence,x,y,w,h\n")

# Rolling MP4 writer: start on detection, stop after tail frames
writer = None
writer_path = None
# While recording, we count down frames and stop exactly at zero.
frames_left_to_record = 0
recording_class = None
input_fps = cap.get(cv2.CAP_PROP_FPS)
if not input_fps or input_fps <= 1.0:
    input_fps = FPS_FALLBACK

print("🚀 Running YOLOv11 Inference... Press Ctrl+C to stop.")

frame_idx = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed frame.")
            break

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (640, 640), swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward()

        # Normalize output shape to (1, N, D)
        # - YOLOv5-style: (1, N, 85) => xywh, obj, 80 classes
        # - YOLOv8-style: (1, 84, N) => xywh + 80 classes (no obj)
        if outputs.ndim == 3 and outputs.shape[1] in (84, 85):
            outputs = np.transpose(outputs, (0, 2, 1))

        # Do not initialize writer here; we start recording only upon detections

        rows = outputs.shape[1]
        dims = outputs.shape[2]
        boxes = []
        confidences = []
        class_ids = []

        fw, fh = frame.shape[1], frame.shape[0]
        for i in range(rows):
            row = outputs[0, i]

            if dims == 84:
                # YOLOv8 head: [cx,cy,w,h, class0..class79]
                scores = row[4:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
            elif dims >= 85:
                # YOLOv5 head: [cx,cy,w,h, obj, class0..class79]
                obj = float(row[4])
                if obj < CONFIDENCE_THRESHOLD:
                    continue
                scores = row[5:]
                class_id = int(np.argmax(scores))
                conf = float(obj * scores[class_id])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
            else:
                # Unknown format
                continue

            if class_id not in ALLOWED_CLASS_IDS:
                continue

            cx, cy, bw, bh = row[0:4]
            x = int((cx - bw / 2.0) * fw / 640.0)
            y = int((cy - bh / 2.0) * fh / 640.0)
            w = int(bw * fw / 640.0)
            h = int(bh * fh / 640.0)

            # Clip to frame bounds
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(1, min(w, fw - x))
            h = max(1, min(h, fh - y))

            boxes.append([x, y, w, h])
            confidences.append(float(conf))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )

        # Process detections
        detected_this_frame = False
        chosen_class_id = None
        chosen_conf = -1.0
        if len(indices) > 0:
            for idx in indices:
                i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                conf = confidences[i]
                detected_this_frame = True
                # Choose the highest-confidence detection in this frame for naming
                if conf > chosen_conf:
                    chosen_conf = conf
                    chosen_class_id = class_id

                # Draw detections
                color = (0, 255, 255) if class_id == 0 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cls_name = (
                    COCO_NAMES[class_id]
                    if 0 <= class_id < len(COCO_NAMES)
                    else str(class_id)
                )
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                # Log every detection immediately (include class label)
                log_file.write(
                    f"{frame_idx},{class_id},{cls_name},{conf:.2f},{x},{y},{w},{h}\n"
                )
                log_file.flush()

        # Handle recording lifecycle: start on first detection, stop after fixed frames
        if detected_this_frame and writer is None:
            # Start a new writer on the first detection only
            fh, fw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Build filename: classnameYYYYMMDDHHMM.mp4
            if chosen_class_id is not None and 0 <= chosen_class_id < len(COCO_NAMES):
                recording_class = COCO_NAMES[chosen_class_id]
            else:
                recording_class = "unknown"
            safe_class = recording_class.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            base_name = f"{safe_class}{timestamp}.mp4"
            out_dir = "/ai/bennwittRepos/velocityView/output"
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, base_name)

            # Avoid accidental overwrite if multiple sessions in same minute
            if os.path.exists(path):
                suffix = 1
                while True:
                    alt = os.path.join(out_dir, f"{safe_class}{timestamp}_{suffix}.mp4")
                    if not os.path.exists(alt):
                        path = alt
                        break
                    suffix += 1

            writer = cv2.VideoWriter(path, fourcc, input_fps, (fw, fh))
            if not writer.isOpened():
                writer = None
                print(f"⚠️ Failed to open MP4 writer at {path}")
            else:
                writer_path = path
                frames_left_to_record = TAIL_FRAMES_AFTER_DETECTION
                print(f"🎬 Started recording: {writer_path} (class={recording_class})")

        # If recording, write frame and count down until reaching the limit
        if writer is not None:
            writer.write(frame)
            frames_left_to_record -= 1
            if frames_left_to_record <= 0:
                writer.release()
                print(f"✅ Saved clip: {writer_path}")
                writer = None
                writer_path = None
                recording_class = None

        frame_idx += 1

except KeyboardInterrupt:
    print("🛑 Interrupted by user. Finalizing current recording (if any)...")
    # If interrupted during an active recording, close and report the file path
    if writer is not None:
        try:
            writer.release()
        finally:
            print(f"✅ Saved clip: {writer_path}")
            writer = None
            writer_path = None
            recording_class = None

finally:
    cap.release()
    if writer is not None:
        writer.release()
    log_file.close()
    print("✅ Detection log saved to output/detections_log.csv")
    # The annotated MP4s are saved per detection session in output/ with naming pattern
    # classnameYYYYMMDDHHMM.mp4 (with an optional _N suffix if needed).
