![VelocityView AI Overview](assets/velocityviewlowpoly2.png)

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
| Descriptions | Apple's FastVLM (1.5B/7B) via `transformers`; structured who/what/when/where/why/how JSON from `describe_inference.py` (Gradio UI) |

---

## 📦 Features

- ✅ YOLOv11 vehicle detection (car, bus, truck, motorbike)
- ✅ Speed estimation from static road ticks
- ✅ Speed violation filtering
- ✅ Headless operation (no GUI required)
- ✅ Overlay export of annotated speed data
- ✅ Modular codebase: detection, logging, estimation, filtering, visualization
- ✅ Vision–language vehicle descriptions with Apple's FastVLM

---

## 📁 Repository Structure
```
velocityView/
├── models/
│   └── yolo11n.onnx                # ONNX model for OpenCV DNN
│   └── FastVLM-1.5B/               # (optional) Local FastVLM snapshot
│   └── FastVLM-7B/                 # (optional) Local FastVLM snapshot
├── output/
│   └── recorded_video.avi          # Raw captured video
│   └── detections_log.csv          # Tick-crossing logs
│   └── speed_events.csv            # Speed estimation data
│   └── violations.csv              # Speeds over limit
│   └── velocity_overlay.avi        # Annotated output video
├── record_webcam.py                # Headless video recorder
├── velocity_inference.py           # YOLOv11-based detection + tick crossing
├── speed_estimator.py              # Calculates speeds from tick logs
├── violation_filter.py             # Flags over-speed events
├── video_overlay.py                # Annotates video with bbox + speed
├── describe_inference.py           # FastVLM-based scene/incident descriptions
├── download_fastvlm.py             # Helper to fetch FastVLM locally
└── README.md                       # You are here
```
---

## 🚀 Getting Started
````
bash
# Set up environment
pyenv virtualenv 3.11.11 velocityView
pyenv activate velocityView
pip install opencv-python ultralytics numpy

# Export model
yolo export model=yolov11n.pt format=onnx

# Run detection pipeline
python record_webcam.py
python velocity_inference.py
python speed_estimator.py
python violation_filter.py
python video_overlay.py
````
---

## 📝 Vehicle Descriptions (FastVLM)

`describe_inference.py` adds a multimodal description step powered by Apple's FastVLM vision–language model. Given an image (for example, a cropped frame of a detected vehicle) it produces a structured, detailed description that answers the classic who/what/when/where/why/how, along with a rich summary of the scene.

- What it does: Generates a strict JSON description inside a `json` code block. Keys include: `who`, `gender`, `age`, `ethnicity`, `what`, `when`, `where`, `why`, `how`, `detailed_summary`, `additional`, `confidence`, and `object_list` (with per-object names/descriptions and relative coordinates).
- How it helps: Pairs quantitative speed detection with qualitative context (e.g., “red pickup overtaking on the right near school crosswalk at dusk”).
- Speed awareness: Your pipeline can attach the measured speed to the same record; the JSON’s `additional` field is a good place to include speed, timestamp, and location metadata.

### Run the describer UI
```
python describe_inference.py
```
- Opens a Gradio UI on `http://0.0.0.0:7860` for interactive testing.
- Upload any frame or crop; copy the returned JSON for downstream use.

### Model setup (FastVLM)
- Requirements: `torch`, `transformers`, `gradio`, `huggingface-hub` (see `requirements.txt`).
- Variants: Script defaults to `apple/FastVLM-7B`. For lighter hardware, use `FastVLM-1.5B`.
- Offline/local loading options:
  - Auto-download from Hugging Face when first run (requires network).
  - Or pre-download locally:
    ```
    python download_fastvlm.py   # downloads apple/FastVLM-1.5B to models/FastVLM-1.5B
    ```
    Then edit `MID` and `MODEL_LOCAL_DIR` at the top of `describe_inference.py` to use the `FastVLM-1.5B` directory.

### Pipeline integration
- Capture: When a violation or noteworthy event is found, save a frame and optionally a tight crop of the vehicle.
- Describe: Feed the crop to `describe_inference.py` (batch or service mode) and merge the returned JSON with your speed/time/place metadata.
- Store: Persist a single incident record combining: image path, measured speed, the FastVLM JSON, and links to your overlay video and CSV logs.

### Output shape (example keys)
The model returns a `json` code block containing keys like:
```
{
  "who": "Unknown",
  "what": "Red pickup truck traveling eastbound",
  "when": "Dusk/low light",
  "where": "Two-lane residential road near crosswalk",
  "why": "Appears to be overtaking slower vehicle",
  "how": "Moving at steady pace; right-side overtake",
  "additional": "Measured speed: 38 mph; Limit: 25 mph",
  "object_list": [ { "name": "truck", "x": 0.62, "y": 0.48, "description": "red pickup" } ],
  "confidence": 78
}
```

Notes:
- Sensitive attributes (gender/age/ethnicity) are optional and should be treated as uncertain; the prompt defaults these to "Unknown" when unclear.
- Use responsibly and comply with local laws and privacy guidance.


## 🧭 Next-Level Roadmap: Where VelocityView Goes from Here

VelocityView is just getting started. The road ahead is packed with possibilities — from smarter inference to citizen-facing dashboards.

Here’s where we’re headed:

---

### 🧠 Intelligence Upgrades

- [ ] **Multi-Car ID + Tracking Across Frames**
  - Integrate DeepSORT or IoU tracking
  - Track each vehicle from entry to exit for multi-span precision
- [ ] **Model Compression + Quantization**
  - Export YOLOv11 with TensorRT or OpenVINO for edge inference
  - Optimize for NVIDIA Jetson, Raspberry Pi with Coral TPU, etc.
- [ ] **Dynamic Tick Calibration**
  - Allow users to place ticks via a simple GUI or upload a JSON config
  - Use perspective math + vanishing points to auto-calculate real-world spacing

---

### 📡 Real-Time Integration

- [ ] **RTSP/USB/VideoStream Support**
  - Expand `VideoCapture` to support live camera IP streams and edge devices
- [ ] **Live Violation Detection + Streaming Dashboard**
  - FastAPI + WebSocket dashboard
  - Stream annotated frames, CSV logs, and live violations in real time

---

### 🛂 Enforcement + Justice Layer

- [ ] **Violation Evidence Packager**
  - Save frame snapshots + logs per violation (video clip + CSV + JSON)
  - Build “evidence folders” auto-sorted per event
- [ ] **License Plate Integration**
  - Plug in OpenALPR or custom fine-tuned OCR for plate logging

---

### 🌍 Community & Civic Impact

- [ ] **Citizen Upload Portal**
  - Drop a recorded video, get back: speeds, violations, and visual overlays
- [ ] **Speed Map**
  - Geo-tagged violation mapping + heatmap by zone
- [ ] **Public API**
  - REST endpoints for data sharing with municipalities, researchers, or visualizations

---

### 📦 Developer-Focused Features

- [ ] **Config-Driven Pipeline**
  - YAML-driven run config (`fps`, `tick_spacing`, `violation_threshold`, etc.)
- [ ] **Plugin Architecture**
  - Drop-in Python modules: detection backend, violation rules, output renderer
- [ ] **Command Line Toolkit**
  - `velocity detect --source /dev/video0`
  - `velocity calc --fps 30 --ticks 10`
  - `velocity export --format mp4`

---

### 🤯 Wild Ambitions

- [ ] **Edge-to-Cloud Sync**
  - Run detection locally, sync results to a remote dashboard for aggregation
- [ ] **Embedded Deployment**
  - Docker container optimized for Jetson Nano / Xavier / Orin
- [ ] **Audio Classification Add-on**
  - Use ambient audio to detect drag racing, revving, or burnout behavior

---

## 🌐 Inspiration & Ethos

VelocityView isn't just a surveillance tool — it's **transparent, user-owned public safety tech**.  
By putting smart enforcement in the hands of neighborhoods and developers, we:

- Empower people, not just systems
- Promote open-source public safety
- Build trust through transparency
- Accelerate civic innovation

---

**→ Want to help? Fork, clone, contribute. Let’s make streets safer — one frame at a time.**  

## 🤝 Contributing

Have ideas to improve tracking, make it real-time, or add multi-camera support?

PRs welcome. Let’s build something the world can use.
