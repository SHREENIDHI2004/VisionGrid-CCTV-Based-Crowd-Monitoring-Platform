# VisionGrid: CCTV-Based Crowd Monitoring & Analytics Platform

VisionGrid is a web-based, real-time surveillance and crowd analytics dashboard powered by YOLOv5. Designed for CCTV feeds, security camera networks, and batch video analysis, the platform provides person tracking, crowd density mapping, heatmaps, and suspicious behavior detection (running, loitering) through a glassmorphism web dashboard.

---

## Key Features

### 📷 1. Image Mode (Single Frame Analysis)
*   **Instant Inference**: Drag & drop or browse photos to run person detection.
*   **Confidence Metrics Grid**: Displays total count, average confidence, highest confidence, and lowest confidence levels.
*   **Bbox & Metadata Table**: View a structured table of detected coordinates and individual confidence values.
*   **Blob-Based Downloads**: Safe one-click image download with overlay badges and numbered bounding boxes.

### 📡 2. CCTV Live Mode (Real-Time Streams)
*   **Multi-Input Streaming**: Supports local webcam inputs (`0`, `1`) and RTSP URLs (e.g., `rtsp://admin:pass@192.168.1.50/stream`).
*   **ByteTrack Person Tracking**: Assigns persistent track IDs to individuals and draws fading motion trails behind them.
*   **4x4 Crowd Density Grid**: Overlays a coordinate grid and visualizes density zones using a green-orange-red color code.
*   **Kernel Density Heatmaps**: Renders a live fading heatmap of foot traffic at the click of a button.
*   **Suspicious Activity Alerting**:
    *   `LOITERING`: Detects when a tracked person remains within a small area for more than 8 seconds.
    *   `RUNNING`: Flags individuals moving faster than a speed threshold (90px/s).
    *   `CROWDED ZONES`: Warns when 5 or more people gather in a single grid cell.
    *   **Live Event Logger**: Keeps a scrolling log of safety triggers with timestamps.

### 🎬 3. Video File Mode (Batch File Processing)
*   **File Inference**: Upload video files (`.mp4`, `.mov`, `.avi`) for offline processing.
*   **In-App Progress Bar**: Monitors frame-by-frame progress in real-time.
*   **Annotated Video Downloads**: Downloads the finished, fully annotated `.mp4` file containing tracking paths and count overlays.

---

## Tech Stack
*   **Backend**: Flask (Python), OpenCV, NumPy, Pillow, Ultralytics YOLOv5 (PyTorch)
*   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism design tokens), Javascript (ES6+, Fetch API, MJPEG stream binder)

---

## Project Structure

```text
VisionGrid-CCTV-Based-Crowd-Monitoring-Platform/
├── app/
│   ├── analytics.py          # Core tracking, loiter detection, & heatmap engine
│   ├── server.py             # Flask streaming API & backend routes
│   └── static/               # Client frontend files
│       ├── index.html        # Clean, responsive studio interface
│       ├── style.css         # Dark theme style tokens & layouts
│       └── main.js           # AJAX handlers, canvas rendering, & tab routers
│
├── data/
│   ├── generate_dataset.py   # Synthetic shape dataset generator
│   └── dataset.yaml          # Dataset config path file
│
├── src/
│   ├── setup_yolov5.py       # Helper script to clone/configure YOLOv5 source
│   ├── train.py              # Wrapper for custom training runs
│   ├── evaluate.py           # Verification script on test data
│   ├── detect_video.py       # Offline script for video inference
│   ├── detect_webcam.py      # Offline script for webcam inference
│   └── export.py             # ONNX/TorchScript export utility
│
├── .gitignore                # Pre-configured ignore rules (ignores weights, pycache, temp files)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Getting Started

### 1. Prerequisite Setup
Make sure Python 3.8+ is installed on your machine. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the Flask server:
```bash
python app/server.py
```

### 3. Open the Dashboard
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
> **Note:** On your first run, the server will automatically download `yolov5su.pt` weights (~17.7 MB) from the official Ultralytics release. Subsequent model loads are instantaneous.

---

## Core Analytics Logic Reference

*   **Loitering Detection**: Tracked coordinates are recorded inside a rolling deque. If the delta between the first and last point's timestamps is greater than `8.0 seconds` and the maximum bounding box shift is less than `100px`, the track ID is flagged red as `LOITERING`.
*   **Heatmap Rendering**: CentOS points are drawn as circles with a radius of `28px` onto a dedicated single-channel float array. A standard `0.997` decay multiplier is applied periodically, and the array is converted to BGR via `cv2.applyColorMap` (Jet scale) to blend over the live frame.
*   **Crowd Density Grid**: Computed by mapping the centroid of each bounding box onto a 4x4 division of the frame width and height, counting active track entries in each cell.
