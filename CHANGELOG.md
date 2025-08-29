
## 2025-08-29
# Detailed Changes

## Version Update
- **Version Increment**: Updated `appVersion` from `0.1.1` to `0.1.3`. This reflects minor improvements in code readability and functionality.
- **Timestamp Change**: The last modified timestamp has been updated to reflect recent changes.

## COCO Names List Improvements
- Added inline comments next to each entry in the `COCO_NAMES` array:
  - These comments denote the index of each item which improves readability and maintainability.
  - Example:
    ```python
    "person",  # 0 
    "bicycle",  # 1 
    ````
- This change helps developers quickly reference indices without manual counting, reducing potential errors during future modifications or debugging sessions.

## Allowed Class IDs Modification
- Refined the list of class IDs that are permissible within a residential context (`ALLOWED_CLASS_IDS`).
- Removed some non-essential classes such as airplanes (4), buses (5), trains (6), etc., focusing on more relevant objects like vehicles, animals, and specific items:
   ```python 
   ALLOWED_CLASS_IDS = [0, 1, 2, ... ,39]    ```  This ensures that only pertinent classes are processed during inference tasks related to typical residential scenarios.

## 2025-08-29
# Detailed Changes

## .gitignore Update
- Added `*.mp4` to ignore list to exclude generated video files.

## New Log Files
- **Detections Log**: Created `output/detections_log.csv` to log frame detections with details such as class ID, confidence score, and bounding box coordinates.
- **Recorded Video Log**: Created `output/recorded_video.csv` capturing performance metrics per frame.
  
## Speed Estimator Adjustments
- Updated file paths in `speed_estimator.py`:
  ```python
  INPUT_CSV = '/ai/bennwittRepos/velocityView/output/detections_log.csv'
  OUTPUT_CSV = '/ai/bennwittRepos/velocityView/output/speed_events.csv'
  ```
- Bumped version from `0.0.4` to `0.0.5`

## Velocity Inference Enhancements
### Configuration Updates:
- Changed model path definitions and added a new output path for annotated videos:
  ```python
  MODEL_PATH = '/ai/bennwittRepos/velocityView/models/yolo11.onnx'
  OUTPUT_VIDEO_PATH = '/ai/bennwittRepos/velocityView/output/detections_annotated.mp4'
  FPS_FALLBACK = 24.0 # Default FPS if input fails detection \\```
CONFIDENCE_THRESHOLD lowered from '0.4' to '0.25' for more sensitivity
```
NMS_THRESHOLD remains at '0.4'.
```
ALLOWED_CLASS_IDS defined comprehensively covering relevant COCO IDs.·YOLOv5 vs YOLOv8 handling logic refined based on dimensions of output layers (84 or >=85).·Implemented exception handling around CUDA backend setup; defaults if unavailable.
automatically creates missing directories using os.makedirs().
buffered writing enabled via open(log_path, "a", buffering=1)
detection annotations directly drawn onto frames with OpenCV functions cv2.rectangle() & cv2.putText().
numpy array operations used extensively within loops ensuring performant batch processing across detected objects per frame index iteratively incremented post-processing each loop iteration finalizes current cycle prepends next ahead anticipated continuation until KeyboardInterrupt triggers termination process closing streams releasing resources gracefully including cap.release(), writer.release(), log_file.close()
informational messages printed post-execution indicating successful saves locations respective outputs ('detection_log', 'detections_annotated').
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
