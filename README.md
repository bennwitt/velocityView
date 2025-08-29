# 🚦 VelocityView
Speed Shaming Through |{A,a}✖️{I,i}|  Computer Vision

> Automated AI-Powered Speed Detection & Violation Logging System  
> Real-time vehicle tracking, speed estimation, and violation analysis using YOLOv11 + OpenCV + CUDA

---

## 🧠 Why VelocityView?

Speeding kills. Enforcement lags.

**VelocityView** is an open-source, computer vision-based solution to automatically:

- Detect vehicles using a standard webcam
- Measure their speed based on fixed road tick marks
- Flag and log violations
- Annotate and output video evidence

It's lightweight, portable, and GPU-accelerated — perfect for communities, researchers, and public-sector innovators looking to reduce speeding **without expensive radar hardware or closed vendor ecosystems**.

---

## 🎯 Who It's For

- 🚸 **Community Advocates**: Monitor school zones, residential streets, or high-risk intersections  
- 🏗️ **Civic Technologists**: Build automated public infrastructure  
- 🧪 **Researchers**: Study vehicle behavior and traffic patterns  
- 💻 **AI Engineers**: Learn real-time CV pipelines with CUDA + YOLO + OpenCV

---

## 💡 Use Case: "The Neighborhood Snitch with a Heart"

You're a resident of a neighborhood plagued by speeding. You mount a webcam with a clear FOV of the road, paint white tick marks at known distances, and run VelocityView.

Within minutes:
- Cars are detected.
- Frame deltas are computed.
- Speeds are estimated.
- Violators are flagged, logged, and optionally shamed in overlaid video exports.

---

## 🌍 Social Good

VelocityView is about **open-source transparency and community empowerment**.

We’re putting the tools of enforcement and awareness in the hands of the people — **no radar guns**, **no vendor lock-in**, just AI for public safety.

---

## 🛠️ Technical Overview

| Component | Tech |
|----------|------|
| Detection | [`YOLOv11`](https://github.com/ultralytics/ultralytics) exported to ONNX |
| Inference | `cv2.dnn.readNetFromONNX()` + CUDA backend |
| Tracking | Bounding box + tick-cross logging |
| Speed Calc | Frame delta × known tick spacing × FPS |
| Violation Filter | CSV-based logic over thresholds |
| Overlay | Annotated bounding boxes + speeds (green/legal, red/violator) |
| Video I/O | OpenCV (headless or recorded) |

---

## 📦 Features

- ✅ YOLOv11 vehicle detection (car, bus, truck, motorbike)
- ✅ Speed estimation from static road ticks
- ✅ Speed violation filtering
- ✅ Headless operation (no GUI required)
- ✅ Overlay export of annotated speed data
- ✅ Modular codebase: detection, logging, estimation, filtering, visualization

---

## 📁 Repository Structure
