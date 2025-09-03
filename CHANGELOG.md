
## 2025-09-03
# Update README.md

## Summary of Changes
- Added an entry under the features table in `README.md`:
  - **Descriptions**: Utilizes Apple's FastVLM models (1.5B/7B parameters) via the `transformers` library.
  - Generates structured JSON output detailing violation circumstances (who/what/when/where/why/how).
  - Offers integration with a Gradio UI for ease of use.

## Technical Details
- **FastVLM**: A model architecture developed by Apple designed for efficient language modeling tasks. Here it's employed to generate descriptive text based on video analysis inputs.
- **Transformers Library**: Used to leverage pretrained models such as FastVLM; ensures high performance inference capabilities.
- **Structured JSON Output**:
  - The generated descriptions are formatted in JSON structure providing clarity and machine-readability.
  ```json
  {
      "who": "",
      "what": "",
      "when": "",
      "where": "",
      "why": "",
      "how": ""
  }
  ```
- **Gradio UI Integration**: Facilitates user interaction allowing non-developers to access complex functionalities without needing direct code manipulation or understanding of backend processes.

## 2025-09-03
# Overview
This commit introduces significant enhancements to the VelocityView project by integrating Apple's FastVLM model, which provides vision-language capabilities. This allows for detailed vehicle descriptions alongside existing speed detection.

## Key Changes
- **Vision-Language Integration**: Added support for generating structured descriptions of detected vehicles using the `describe_inference.py` script.
  - Utilizes Apple's FastVLM model to produce JSON-formatted outputs detailing aspects such as 'who', 'what', 'when', and more.
- **Script Updates**:
  - Renamed `velocity_infer.py` to `velocity_inference.py` for consistency with function naming conventions.
  - Introduced new scripts: 
    - `describe_inference.py`: Handles inference requests using the FastVLM model.
    - `download_fastvlm.py`: Facilitates downloading of required model files locally from Hugging Face Hub.
- **Directory Structure**:
  - Added directories under `models/` for storing local snapshots of different versions of the FastVLM model (`FastVLM-1.5B`, `FastVLM-7B`).

## Detailed Usage Instructions
### Vehicle Description Generation with FastVLM
The integration empowers users with a multimodal description step powered by Apple’s vision-language technology, enhancing context awareness in traffic monitoring systems:
```python
git clone <repo-url>
pip install -r requirements.txt # Ensure dependencies are installed properly
python describe_inference.py   # Launches Gradio UI at http://0.0.0.0:7860/
upload_frame_or_crop()         # Test image uploads; copies resulting JSON output
download_fastvlm_model()       # Downloads models if not already present locally
```						        	      	     	            	                    	           ## Model Setup & Variants   * Default is set to use larger capacity variant (Fast VMLM –7 B). For limited resource environments switch configuration parameters found at top lines within ‘describe inference py’ file.* Local download option ensures robustness against connectivity issues when setting up pipeline first time around or in isolated environments without internet accessibility.# Output Format Example* The generated JSON includes keys like:`{“who”: “Unknown”, “what”: “Red pickup truck traveling eastbound”,…}`# Important ConsiderationsSensitive attributes should be handled cautiously respecting privacy guidelines & regulations applicable jurisdictionally.

## 2025-09-03
# Summary of Changes

## .gitignore Updates
- **Added**: `models/FastVLM*` to ignore any files related to models in this directory.

## describe_inference.py Modifications
### Metadata & Versioning:
- **Updated** modification timestamp.
- **Incremented** `appVersion` from `0.0.1` to `0.0.7`, indicating significant changes including feature enhancement and bug fixes.

### Model Configuration:
- **Model ID Change**: Updated from "apple/FastVLM-1.5B" to "apple/FastVLM-7B" reflecting an upgrade in the underlying language model used for inference.

### New Functionality:
#### `_resolve_local_model_dir`
  - A new utility function designed to locate directories containing a configuration file (`config.json`).
  - This supports flexible local path resolutions accommodating various directory structures typical in Hugging Face snapshots.
  \\```python
def _resolve_local_model_dir(base_dir: str) -> str | None:
```
and so forth...

#### Image Description Logic Enhancements:
  - Transitioned from simple text prompts to structured JSON outputs using Gradio's interface capabilities.
  - The response now strictly adheres to a predefined schema enclosed within code blocks ensuring consistency and ease of parsing by downstream applications or services.
n\\```json{"who": "Unknown", ...}
n\\```
instructions provided ensure that each field is addressed correctly even when data is uncertain (e.g., use of 'Unknown' for sensitive attributes).
n\# Inference Generation Adjustments:\\
nMax token length considerations were added ensuring compatibility across various contexts sizes with dynamic calculations based on existing sequence lengths.nnThe decoding process was refined allowing more robust handling of sequences avoiding truncation issues.nnFinally set server_name parameter during app launch enabling broader accessibility options via network interfaces.n

## 2025-09-03
# Changes Overview

## Description
This commit introduces enhancements to the `describe_inference.py` script by integrating a mechanism to utilize locally cached models, specifically targeting the FastVLM-1.5B model. Additionally, a new utility script, `download_fastvlm.py`, is provided to facilitate the downloading of this model into a designated directory.

## Key Modifications
### describe_inference.py:
- **Import Statement**: Added an import statement for `os` module which is necessary for handling file paths.
  ```python
  import os
  ```
- **Constants Definition**: Introduced `MODEL_LOCAL_DIR` constant that defines the path where local models are stored.
  ```python
  MODEL_LOCAL_DIR = os.path.join("models", "FastVLM-1.5B")
  ```
- **Model Loading Logic**:
  - Updated logic in `load_model()` function to check if a local copy of the model exists before attempting to download from Hugging Face's repository.
  - Set up conditional paths (`source_path`) based on directory availability with preference given to existing local copies.
  - Modified calls for both tokenizer and model instantiation with parameters such as `cache_dir` and `local_files_only` set appropriately based on source location (local vs remote).
   \\[Example Code]: \\​​​…색이사이드; \\[end]\\[start](code)
u0028code)tokenizer = AutoTokenizer.from_pretrained(
sourc…;
u0028/u2026); \newline; ; ; ; ,trust_remote_code=True,
cach…;
n_source_path == MID else None,
l_local_files_onl…;
m_source_path == MODEL_LOCAL_DIR,
a);
u0028/u2026); \newline; # Model Instantiation\newline; );model = AutoModelForCausalL…;
torched_dtype=dtype,
d_device_map="auto",
t_trust_remote_co…;
d_cache_dir=MODEL_LOC... [end code]) ### New Script: download_fastvlm.py:

#### Purpose
Provides functionality to download and store specific versions or snapshots of machine learning models locally via Hugging Face Hub API integration.

#### Functionality Overview
* Implements main function executing snapshot downloads utilizing 'huggingface-hub' library functions while ensuring target directories exist prior execution through use built-in 'os' package capabilities like makedirs().*
* Allows ignoring non-essential files during downloads optimizing space usage when specified within ignore_patterns argument parameter.*

## 2025-09-03
# New Features

## Gradio Interface Implementation
- **Script**: Added `describe_inference.py`
  - Provides an interactive interface for users to upload images and receive detailed textual descriptions.
  - Utilizes Hugging Face's Transformers library to load the `apple/FastVLM-1.5B` model and tokenizer.
  - Automatically selects device (GPU or CPU) based on availability for efficient inference.

## Key Functions
- **load_model()**
  - Loads tokenizer and language model only once at startup, reducing overhead during runtime.
  - Determines data type (`float16` or `float32`) based on GPU availability for performance optimization.

- **describe_image(img: Image.Image) -> str**
  - Accepts a PIL Image object as input and returns a string description generated by the model.
  - Constructs input sequences with special handling of `<image>` tokens where vision features are inserted into text prompts.

## Application Launching
- **main() function**
  - Initializes and launches a local Gradio server at default address `http://localhost:7860/`, providing an accessible web UI for interaction with the image description service.

# Dependencies Management & Requirements File Updates (requirements.txt):

* Added comprehensive list of dependencies required to run the application, ensuring all necessary packages are installed:
   * Libraries such as torch, transformers, gradio among others ensure compatibility and functionality of both frontend (Gradio) & backend (model inference).
   * Specific versions locked down like accelerate==1.10.1, gradio==5.44.1 etc., facilitate consistent environment setup across different machines.

## 2025-09-02
# Changes Overview

## Version Update
- **Updated `appVersion`**: Changed from `0.3.17` to `0.3.25`, reflecting significant updates in model parameters.
- **Modification Timestamp**: Updated last modified date to reflect changes.

## Parameter Adjustments
- **Confidence Threshold**:
  - Increased from `0.54` to `0.71`. This adjustment aims to reduce false positives by requiring higher confidence levels before detections are considered valid.
  
```python
CONFIDENCE_THRESHOLD = 0.71
```
- **Tail Frames After Detection**:
  - Added comment clarification, noting that this parameter creates clips of approximately ~5 seconds duration post-detection.
  
```python
TAIL_FRAMES_AFTER_DETECTION = 150 # ~5 second clips
```

## COCO Class Names Optimization:
- Trimmed the list of detectable objects down significantly, focusing only on those most likely encountered in a residential environment or required by the application logic.
- Removed redundant and unlikely classes such as "airplane", "bench", etc., which are less relevant in typical use cases of this project.
 ```python 
person, bicycle, car, motorcycle,
bus, truck, boat,
dog,...
surfboard... scissors...
note: Indices updated accordingly...
also reflected in ALLOWED_CLASS_IDS array update...
and comments adjusted for clarity...
to ensure accurate mapping between indices and class names...
and enhance code maintainability and readability... ```
in summary these changes aim at improving detection performance while maintaining efficient processing...

## 2025-09-01
# Detailed Commit Information

## Overview
This commit introduces a change in the configuration of the velocity inference system by adjusting the number of frames recorded after an initial detection and updating the application version.

## Changes Made
- **Version Update**
  - The application version was incremented from `0.3.16` to `0.3.17`. This signifies a minor update, reflecting changes that might affect behavior but do not introduce breaking changes.
- **Tail Frame Adjustment**
  - Modified `TAIL_FRAMES_AFTER_DETECTION`:
    ```python
    # Before:
    TAIL_FRAMES_AFTER_DETECTION = 300
    
    # After:
    TAIL_FRAMES_AFTER_DETECTION = 150
    ```
   
### Implications of Change:
- **Performance Impact:** Reducing this parameter will potentially decrease memory usage and processing time for video segments where detections occur frequently, as fewer frames are retained post-detection.
- **Use Case Considerations:** Ensure that reducing tail frames aligns with expected use cases, particularly if subsequent frame analysis is critical post-initial detection.
- **Testing Requirements:** Validate that this change does not adversely impact downstream processes or analytics relying on extended frame data following an object detection event.

## 2025-09-01
# Image Update in README

## Overview
This commit updates the main visual representation of VelocityView AI within the `README.md` by replacing:
- **Old Image**: `assets/velocityviewlowpoly.png`
- **New Image**: `assets/velocityviewlowpoly2.png`

## Changes Made
- **README.md**: Changed markdown image link.
  ```diff
  -![VelocityView AI Overview](assets/velocityviewlowpoly.png)
  +![VelocityView AI Overview](assets/velocityviewlowpoly2.png)
  ```
- **Binary File Updates**:
  - The previous image file, located at `assets/velocityviewlowpoly.png`, has been replaced as indicated by differing binary files.
  - Introduced a new binary file named `assets/velocityviewlowpoly2.png`, which serves as the latest graphical asset for our project documentation.

## Rationale Behind Change
Updating visuals is crucial for maintaining up-to-date and accurate representations of our project's capabilities. This change ensures that users are greeted with an improved and possibly more informative graphic when accessing our repository's main page.

## 2025-08-31
# Update README Image

## Summary
- **Replaced Image**: The existing `velocityviewsq.png` has been replaced by `velocityviewlowpoly.png`.
- **New Asset**: Added `velocityviewlowpoly.png` as a new asset under the same directory.

## Technical Details
### Changes Made:
1. **README.md Update**
   - Changed the reference from:
     ```markdown
     ![VelocityView AI Overview](assets/velocityviewsq.png)
     ```
   - To:
     ```markdown
     ![VelocityView AI Overview](assets/velocityviewlowpoly.png)
     ```
   This updates the displayed image on GitHub's markdown renderer to use a more stylized graphic.
2. **Asset Addition**
   - Introduced a new binary file, `assets/velocityviewlowpoly.png`, which is now tracked in version control. This file is intended for usage within our project documentation and replaces its predecessor without altering any functional code components.

## Rationale for Change:
The decision to switch images was driven by an aesthetic update that reflects modern design sensibilities while maintaining clarity and relevance of information being conveyed through visuals.

## 2025-08-31
# Updated Image Asset: `velocityviewsq.png`

## Overview
The commit involves updating an image asset, specifically `velocityviewsq.png`. The update replaces the previous binary file with a new one.

### Details:
- **File Path**: `assets/velocityviewsq.png`
- **Previous Commit Index**: `f71ee9d`
- **New Commit Index**: `adba8d9`

## Technical Explanation:
Binary files differ in their data structure from text-based files and therefore don't show line-by-line differences in diffs. Instead, they are replaced entirely when modified. In this scenario, we have:

- A complete replacement of the old binary data within the PNG file.

### Impact on Project:
This change primarily affects areas of your project where this specific image is utilized. It's crucial for maintaining up-to-date visual elements across user interfaces or documentation that reference this image.

### Recommendations for Reviewers:
1. Verify that the updated image meets design specifications and requirements.
2. Check integration points where this asset is used to ensure compatibility and correctness after update.

## 2025-08-31
# Update Overview

## Changes Made:
- **README.md**: 
  - Introduced an image at the beginning using markdown syntax to embed visuals effectively.
  - The image is located in `assets/velocityviewsq.png` and provides users with immediate context upon opening the README.
- **Assets Directory**:
  - Added a new file `velocityviewsq.png` under `assets/`. This binary addition ensures that all images are stored systematically within one directory, maintaining project organization.

## Technical Details:
- The markdown syntax used to include this image is as follows:
  ```markdown
  ![VelocityView AI Overview](assets/velocityviewsq.png)
  ```
- Markdown's image embedding allows for linking local files directly within documentation, which can be beneficial when aiming for self-contained project directories without external dependencies.
- Binary files like images do not display diffs in text form due to their non-text nature but are crucial for enhancing user interaction through visual elements.

## Considerations:
- Ensure that any additional assets added in future commits follow this organizational pattern by residing under appropriate directories (e.g., 'assets/') ensuring neatness and accessibility across team members or open-source contributors.

## 2025-08-31
# Details of Changes

## Summary
The following files have been removed:
- **`output/clips_log.csv`**: Previously stored metadata about video clips.
- **`output/detections_log.csv`**: Contained detection results for frames processed.
- **`output/recorded_video.csv`**: Logged performance metrics of video recording operations.

## Technical Explanation
### Purpose of Removal
These files were likely used for debugging or development purposes, capturing detailed logs that are no longer necessary in production. Removing them can lead to several benefits:
- **Reduced Disk Usage**: Each file occupied space without providing ongoing value, especially in long-running systems where logs accumulate over time.
- **Simplified File Management**: Fewer outputs mean easier navigation through directories and less clutter when accessing relevant data.

### Impact on System Functionality 
This change does not affect core functionalities if these logs were only supplementary. However, ensure that any dependent systems or scripts referencing these CSVs are updated accordingly to avoid errors due to missing files.

### Code Diff Highlights 
diff --git a/output/clips_log.csv b/output/clips_log.csv
deleted file mode 100644...
diff --git a/output/detections_log.csv b/output/detections_log.csv
deleted file mode 100644...
diff --git a/output/recorded_video.csv b/output/recorded_video.csv
deleted file mode 100644...
Each diff segment indicates complete removal with lines transitioning from content-filled (indicated by '-') to empty ('+') states.

## 2025-08-29
# Detailed Changes

## Updated .gitignore
- **Removed**: The line `*.onnx`, which previously ignored all files with the `.onnx` extension.
  - This change is necessary to track specific ONNX files within our repository, allowing for better version control of these machine learning models.
  
## Added YOLO Model File
- **File Added**: `models/yolo11n.onnx`
  - This is a binary file representing an Open Neural Network Exchange (ONNX) format model.
  - The addition of this specific model suggests its importance in our project, likely related to neural network tasks such as object detection or classification using YOLO architecture.

# Technical Considerations
- **ONNX Files**: Typically used for sharing models between different frameworks. By tracking them in Git, we ensure that any updates or improvements made to these models can be documented and reverted if necessary.
- **Binary Files in Repos**: While generally avoided due to size considerations and lack of diff support, including essential binary files like trained ML models can be critical for reproducibility and deployment consistency.

## 2025-08-29
# Detailed Changes

## New Features
- **Clip Logging**: Introduced a new `clips_log.csv` file that logs details about each recorded clip.
  - Metadata includes:
    - Class name
    - Filename of the saved clip
    - Start and end timestamps in ISO format
    - Total frames written during the session
    - FPS reported by OpenCV vs. FPS measured based on real-time duration calculation.
- **Timestamp Precision**: Enhanced timestamp precision in filenames from minute-level to second-level granularity for better uniqueness.
- **Frame Accounting**: Added counters for:
  - Frames captured during active detection sessions (`frames_captured_current_clip`)
  - Frames actually written/saved (`frames_saved_current_clip`).
 
## Code Structure Enhancements
- Wrapped critical sections with try-finally blocks to ensure resources are properly released even if exceptions occur (e.g., releasing video writer objects).
- Utilized Python's `datetime.now().isoformat(sep=' ')` for consistent timestamp formatting across logs.
 
## Technical Implementation Details:
```python
def record_frame(writer, frame):
   """
describes how each frame is processed before being logged or saved."""
writes(frame)
saves_to_csv()
increments_counter()
stops_when_limit_reached()```
persisted_states = [writer_path, current_clip_start_time]
buffered_operations = [flush(), release()]```
time_calculations = [(end_time-start_time).total_seconds(), max(1e-06)]```
timestamp_formatting = datetime.now().strftime("%Y%m%d%H%M%S")```

## 2025-08-29
# Changes Overview

## New Features
- **CSV Logging**: Introduced `detections_log.csv` to log object detections with details such as frame number, class ID, confidence score, and bounding box coordinates.
  
  ```csv
  frame,class_id,class_name,confidence,x,y,w,h
  ...
  ```
- **Version Update**: Incremented application version from `0.2.11` to `0.3.13`. This reflects significant changes including improved functionality in handling detection logs and recording logic.

## Video Recording Logic Improvements:
1. **Start/Stop Mechanism**:
   - Initiates video recording upon detecting the first object of interest (`detected_this_frame`) instead of resetting on each subsequent detection.
   - Stops writing frames after a predetermined number (`TAIL_FRAMES_AFTER_DETECTION = 300`) rather than tailing until no more objects are detected within a sliding window.
2. **File Naming & Handling**:
   - Generates filenames using the format: `{class_name}{timestamp}.mp4`, ensuring uniqueness even if multiple files are created within the same minute via suffix incrementation when needed.
3. **Code Refactoring**:
   - Replaced variable names for clarity (e.g., `frames_tail_left` renamed to `frames_left_to_record`).
   	```python
	# Old code snippet example:
def handle_old_logic():
detected_this_frame = check_detection()
writertail_frames_left -= 1if not detected_this_frame or writertail_frames_left <= 0:writernone```
to...	```python
def handle_new_logic():	detected_this_framecheck_detection()if detected_this_framewriterNone:start_writer()elseframes_left_to_record-=1	```	# Benefits	This change reduces redundant I/O operations by preventing frequent stopping/starting based on fluctuating detections during active sessions.# ImpactThese enhancements streamline processing efficiency by reducing CPU load associated with constant file writes while maintaining comprehensive tracking through structured data logging.

## 2025-08-29
# Enhancements in Video Detection

The recent changes bring significant improvements to the way detections are logged and videos are recorded.

## Key Changes:
- **Rolling Recording Logic**: Now, video recordings start only upon detecting specified classes, continuing until a specified number of frames after the last detection.
- **Dynamic File Naming**: Recorded clips are named using a pattern that includes the class name and timestamp, ensuring unique filenames even within consecutive sessions.
- **Removed Static Logging File**: The previously used `detections_log.csv` has been removed in favor of more dynamic handling directly within `velocity_inference.py`.

## Detailed Breakdown:
### Code Structure Adjustments:
1. **Initialization Improvements**
   - Removed premature MP4 writer initialization; it now starts with detections only.
   - Added variables like `writer_path`, `frames_tail_left`, and `recording_class` for managing state across frames.
2. **Detection Processing Loop**
   - Each frame checks for new detections; if found, resets tail counter and potentially initiates a new recording session with an appropriately generated filename using class names (`COCO_NAMES`) converted into safe string formats (e.g., replacing spaces).
3. **File Management & Safety Checks:**		  		  		  		  		  	     - Ensures no overwriting by checking existing files before starting a new one — appending numeric suffixes if necessary (`_N`).	     - Handles interruptions gracefully by finalizing active recordings properly when user interrupts via Ctrl+C or similar methods.

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
