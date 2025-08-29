
## 2025-08-29
# Video Overlay Enhancements

This commit introduces several significant improvements to the `video_overlay.py` script:

## Changes Made
- **Version Update**: Incremented `appVersion` from 0.0.6 to 0.1.11.
- **File Format Change**: 
  - Changed input (`recorded_video`) and output (`velocity_overlay`) files from `.avi` to `.mp4`, reflecting modern usage trends.
- **Dynamic FPS & Resolution Handling**:
  - Introduced automatic detection of frame width, height, and frames per second (FPS) using OpenCV properties:
    ```python
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = in_fps if in_fps and in_fps > 1.0 else FPS_FALLBACK
    out_size = RESOLUTION if RESOLUTION else (in_w, in_h)
   ```
   - This ensures that the script adapts to various input videos without manual configuration unless explicitly overridden.
- **Codec Update**: Switched from 'XVID' codec used by AVI format to 'mp4v', suitable for MP4 files:
   ```python
   fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, out_size)	```						     	     	     	         	       n e t o u r e s i l i o n a c h i v e d w h e n c a p t u r i n g f r a m es . T h is s“️ Failed to open MP writer your OpenC may lack support” ) ````      # Resize if desired output size differs+if (frame.shape[1], frame.shape[0]) != out_size:+frame=cv.resize(frame,out_size)out.write(frame)+frame_idx+=1

## 2025-08-29
# Changes in `detections_log.csv`

- **Added Class Name**: The CSV header now includes a `class_name` column, which provides more context by associating each detection with its respective category.
  ```diff
  -frame,class_id,confidence,x,y,w,h
  +frame,class_id,class_name,confidence,x,y,w,h
  ```
- **Improved Logging Format**: Each logged entry now records the class name alongside other attributes:
  ```python
  log_file.write(f"{frame_idx},{class_id},{cls_name},{conf:.2f},{x},{y},{w},{h}\n")
  ```

# Adjustments in `velocity_inference.py`

- **Version Update**: Incremented application version from `0.1.4` to `0.2.9`, reflecting significant improvements.
- **Confidence Threshold Change**: Increased from `0.25` to `0.54`, enhancing object detection reliability by filtering out less certain predictions:
  ```diff
  -CONFIDENCE_THRESHOLD = 0.25
  +CONFIDENCE_THRESHOLD = 0.54
  ```
- **CSV Header Modification**: Updated logic for writing headers if needed:
    ```python log_file.write("frame,class_id,class_name,confidence,x,y,w,h\u00a0\u00a09``)```		-	# Log every detection immediately (include class label) # Log every detection immediately (include class label)
+                log_file.flush()

# Impact of Changes

These modifications ensure that each entry in our logs is not only accurate but also enriched with categorical labels that can assist further analysis or debugging processes._Adjusting_ the confidence level helps maintain high-quality detections by minimizing false positives.

## 2025-08-29
# Update Details

## Detections Log Enhancements
- **Added Entries**: Multiple new rows have been appended to `detections_log.csv`, expanding the dataset significantly.
- **Data Structure**:
  - Each entry consists of frame number, class ID, confidence score, and bounding box coordinates (x, y, w, h).
  - Example of an added entry: `19,15,0.36,0,208,141,167`

## Code Changes in velocity_inference.py
- **File Metadata Update**:
  - Updated last modified timestamp from `2025-08-29 10:41:32` to `2025-08-29 10:48:36`.

```python
timestamp = "2025-08-29T10:48:36"
```

### App Version Increment:
The application version has been updated as follows:
```python
appVersion = "0.1.4" # Previously "0.1.3"
```
This reflects modifications made within this commit cycle.

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
