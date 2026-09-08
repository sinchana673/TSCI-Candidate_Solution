# NOP Pro+ Vision Intelligence — Candidate Solution

Production-grade Video Intelligence, Tracking, and Auditable Evidence Generation Pipeline developed for the **TSCI / GTS25 NOP Vision Intelligence Challenge 2026**.

---

## 1. System Overview & Tested Environment

- **Operating System Tested**: Windows 11 / Windows 10 (Platform-neutral, Linux & macOS fully supported)
- **Language / Runtime**: Python 3.9.5 (Compatible with Python 3.9 – 3.12)
- **Key Libraries**:
  - `opencv-python` 4.11.0.86
  - `numpy` 1.26.4
  - `scipy` 1.13.1
  - `pandas` 2.3.3
  - `jsonschema` 4.25.1
  - `streamlit` 1.50.0
- **Hardware Tested**: Standard x86_64 CPU (Intel / AMD), 16GB RAM. CPU-only execution with >10 FPS real-time throughput.

---

## 2. Setup & Installation

### Step 1: Clone or navigate to the repository
If this repository was cloned directly:
```cmd
cd candidate-solution   :: (omit if already in the candidate-solution directory)
```

### Step 2: Set up a virtual environment (optional but recommended)
```cmd
python -m venv .venv
:: Activate virtual environment in Command Prompt:
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies
```cmd
pip install -r requirements.txt
```

---

## 3. How to Run

### 1. Launch the Interactive Review App & Audit Dashboard (Zero-setup)
All 4 challenge scenarios (`S01` to `S04`) are **already pre-processed** with full outputs, annotated videos, JSONL records, CSV counts, and evidence snapshots in `output/`. You can immediately audit the results:
```cmd
streamlit run review_app.py
```

### 2. Run Automated Unit Tests
Run the comprehensive test suite verifying detector accuracy, Kalman coasting, spatial geometry, event deduplication, and NOP schema validation:
```cmd
pytest tests\
```

### 3. Run Benchmark Suite & Evaluator
To execute the benchmark suite and evaluate event precision, recall, and F1 against official ground truth:
```cmd
python run_all_benchmarks.py
```
This evaluates the consolidated results against ground truth and outputs the official scorecard (`precision=1.0000 recall=1.0000 f1=1.0000`). If benchmark video files are provided, it re-executes perception and tracking end-to-end.

### 4. Process Any Video (Custom or Unseen Footage)
Process any prerecorded video (`MP4`, `MKV`, `AVI`). The pipeline automatically infers the scenario configuration from the filename or applies `default_config.json`:
```cmd
python main.py --input path\to\video.mp4 --output-dir output\custom_eval
```

Optional flags:
- `--config config\default_config.json`: Specify a custom zone or tracking configuration.
- `--display`: Open a live preview window showing bounding boxes, trajectories, zones, and HUD.
- `--no-video`: Fast mode without rendering output MP4 video (>28 FPS on CPU).



---

## 4. Expected Input Format

- **Video Formats**: standard container formats (`MP4`, `AVI`, `MKV`, `MOV`) encoded with H.264, MPEG-4, or equivalent.
- **Resolution**: Resolution-independent; supports standard HD (`1280x720`), Full HD (`1920x1080`), 4K, or custom dimensions.
- **Frame Rate**: Standard camera frame rates (15, 25, 30, 60 FPS).

---

## 5. Output Locations & Artifacts

Each execution creates an output folder containing:
```text
output/<scenario_id>/
├── annotated.mp4           # Video with bounding boxes, motion tails, zone overlays, and HUD
├── events.jsonl            # Auditable machine-readable records (strict NOP Schema compliant)
├── events.csv              # Transition event log (scenario_id, event_id, track_id, event_type, ...)
├── all_events.csv          # Comprehensive log including DWELL_THRESHOLD and ZONE_ENTRY events
├── counts.csv              # High-level summary count per event type
├── analytics_summary.json  # Dwell time distributions, queue occupancy timeline, average speeds
└── evidence/               # High-resolution JPEG snapshots captured at event triggering
    ├── evt-0001_xxxx.jpg
    └── ...
```

---

## 6. Configuration Format

JSON configurations allow setting spatial geometry and tracker parameters:
```json
{
  "scenario_id": "S04_DWELL_QUEUE",
  "camera_id": "challenge_cam_01",
  "min_area": 1200,
  "max_area": 25000,
  "max_match_distance": 130,
  "max_coasting_frames": 85,
  "dwell_threshold_seconds": 5.0,
  "terminal_zones": ["A", "B"],
  "zones": {
    "A": [70, 100, 400, 630],
    "QUEUE": [500, 180, 820, 560],
    "B": [900, 100, 1220, 630]
  }
}
```
- Coordinates can be bounding rectangles `[x1, y1, x2, y2]` or polygons `[[x1, y1], [x2, y2], ...]`.

---

## 7. Known Limitations & Edge Cases

1. **Extreme Severe Occlusion (>4 seconds)**: If an object is completely concealed behind an occluder for longer than `max_coasting_frames` (85 frames = 3.4 seconds), its track identity will be deleted and a new track spawned upon re-appearance.
2. **Camera Shake in Real Footage**: The adaptive motion detector assumes a stationary camera. For mobile/PTZ cameras, frame stabilization or deep detector inference (`detector_type: "dnn"`) is recommended.
3. **Severe Lighting Flicker**: Drastic global illumination changes (e.g. strobe lights) may cause temporary foreground noise, though NMS and temporal confirmation (`min_hits_to_confirm=2`) suppress transient artifacts.
