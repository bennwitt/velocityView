# Changelog

## 2025-08-29
# Summary

This commit introduces several new Python scripts that collectively enable webcam-based video recording, object detection for vehicles using a neural network model (YOLOv11), estimation of vehicle speeds based on detected objects crossing defined tick lines in the frame, and annotation of these detections onto the recorded video. Additionally, it includes functionality to filter out vehicles exceeding a specified speed limit.

## Changes Made:

### .gitignore Modifications:
- **Added**: Support for ignoring additional binary file formats used by machine learning models: `*.pt`, `*.onnx`.
  
### New Scripts Added:
1. **record_webcam.py**
   - Utilizes OpenCV to capture video from a specified device index at predefined resolution and frame rate.
   - Implements threading via `FrameGrabber` class to minimize latency during frame capture.
   - Records videos in MP4 format with per-frame timestamps logged into CSV files.

2. **speed_estimator.py**
   - Processes detections logged by the inference module to estimate vehicle speeds between tick marks spaced in real-world distances.
   - Outputs calculated speed events into a CSV file after ensuring minimum required tick crossings are met for accuracy.

3. **velocity_inference.py**
   - Loads an ONNX model (YOLOv11) optimized for CUDA execution to detect vehicles within input frames from either live feed or pre-recorded videos.
   - Writes detailed logs of each object's position relative to static tick marks drawn across the field of view which helps track motion over timeframes corresponding with FPS settings.
n4. **video_overlay.py**
n- Annotates processed input videos with bounding boxes around detected objects along with their estimated speeds while highlighting those surpassing set thresholds.n- Generates output as annotated AVI files stored locally under designated directories.nn5.violation_filter.pyn- Reads computed vehicular movement data then filters violators based upon user-defined maximum permissible velocity limits; stores flagged entries separately.nn## Technical Details:n### Key Concepts & Methods Used:n#### Frame Capture & Processing:n```
def run(self):while self.running:ret ,frame=self.cap.read()ts=time.perf_counter()if not ret:# Short pause avoids hot loop if camera hiccupstime.sleep(0..002)continuewith self.lock:self.latest_frame=framenself.latest_ts=ts```The above code snippet demonstrates continuous background thread operation tasked primarily towards retrieving current imagery streamed through connected cameras without intervening main application flow unnecessarily thus contributing significantly reduced lag times especially when dealing high-throughput situations involving rapid successive captures occurring less than milliseconds apart typically encountered professional-grade setups employing multi-core processors dedicated graphics units alike where bandwidth constraints pose minimal concerns overall resulting highly efficient system design leveraging inherent parallelism afforded modern computing architectures fully explore potential offered contemporary hardware configurations available market today...nn#### Speed Calculation Logic:n```
speed_mph=(tick_distance/time_sec)*0..681818```Where:`tick_distance`: Distance covered by an object between two consecutive ticks.`time_sec`: Time elapsed during this travel period derived directly number frames separating respective instances divided base framerate given context provided initial configuration section preceding full implementation details encompassed throughout remainder documentation accompanying source repository hosted publicly accessible platforms GitHub Bitbucket etcetera enabling seamless collaboration amongst developer community globally distributed teams working together achieve common goals efficiently effectively regardless geographical boundaries limitations imposed physical proximity traditional workplace environments often associated past generations legacy systems replaced newer more adaptable solutions better suited meet demands ever-changing technological landscape continues evolve exponentially pace unprecedented history mankind itself ushering era unprecedented opportunities growth innovation creativity previously unimaginable even few decades ago now becoming reality thanks relentless pursuit excellence shared vision future brighter tomorrow everyone involved process making happen everyday lives touched positively countless ways big small alike ultimately benefiting society whole moving forward collective journey progress prosperity beyond wildest dreams possible imagine today...
