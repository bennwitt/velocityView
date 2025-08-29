# Last modified: 2025-08-28 16:18:53
appVersion = "0.0.3"
# velocity_infer.py - Phase 1
import cv2
import numpy as np
import time
import os

# Config
MODEL_PATH = "/ai/bennwittRepos/velocityView/models/yolo11n.onnx"
VIDEO_INPUT = 0  # Use 0 for webcam, or path to video
TICK_LINES = [400, 350, 300, 250]  # y-pixel positions for static tick marks
TICK_SPACING_FT = 10  # Distance between tick lines in real world
FPS = 24.0
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
LABELS = ["car", "truck", "bus", "motorbike"]

# YOLOv11 class IDs for vehicles (COCO)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Load network
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Video setup
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise RuntimeError("Failed to open video input")

# Output dir
os.makedirs("output", exist_ok=True)
log_file = open("/ai/bennwittRepos/velocityView/output/detections_log.csv", "w")
log_file.write("frame,tick_id,class_id,confidence,x,y,w,h\n")

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

        rows = outputs.shape[1]
        boxes = []
        confidences = []
        class_ids = []

        for i in range(rows):
            row = outputs[0, i]
            confidence = row[4]
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(np.argmax(row[5:]))
                if class_id in VEHICLE_CLASS_IDS:
                    cx, cy, w, h = row[0:4]
                    x = int(cx - w / 2) * frame.shape[1] // 640
                    y = int(cy - h / 2) * frame.shape[0] // 640
                    w = int(w * frame.shape[1] / 640)
                    h = int(h * frame.shape[0] / 640)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )

        # Draw tick lines
        for i, y in enumerate(TICK_LINES):
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 1)

        # Process detections
        for idx in indices:
            i = idx[0]
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            conf = confidences[i]
            center_y = y + h // 2

            # Check for tick crossings
            for tick_id, tick_y in enumerate(TICK_LINES):
                if abs(center_y - tick_y) < 5:
                    log_file.write(
                        f"{frame_idx},{tick_id},{class_id},{conf:.2f},{x},{y},{w},{h}\n"
                    )

        frame_idx += 1

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

finally:
    cap.release()
    log_file.close()
    print("✅ Detection log saved to output/detections_log.csv")
