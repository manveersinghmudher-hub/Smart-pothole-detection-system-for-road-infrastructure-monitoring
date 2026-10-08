# Smart Pothole Detection System (SPDS) 🚗🛣️

**An AI-Powered Pothole Detection Pipeline with False-Positive Filtering.**

This project was built to accurately detect potholes in real-time while solving a critical flaw in standard object detection: _false positives on non-drivable surfaces (like shadows on buildings or off-road terrain)._

## 🧠 How It Works (The Secret Sauce)

Instead of relying on a single object detector, our pipeline uses a **Two-Stage Hybrid Architecture**:

1. **D-FINE Object Detection**: Identifies potential potholes with high recall.
2. **PIDNet-S Semantic Segmentation**: Maps the drivable road surface in real-time.
3. **Spatial Verification Layer**: Any pothole detected _outside_ the PIDNet road mask (e.g., in the sky, on a building, or off-road) is immediately rejected.
4. **Temporal Tracking**: Uses IoU tracking across video frames to prevent detections from flickering.

---

## 🚀 Quick Start Guide

### Step 1: Set up the project

From the project root, open a terminal and run:

```bash
python setup.py
```

Wait for setup to finish installing dependencies and preparing the model files.

### Step 2: Start inference

Run `run_inference.bat`. When the terminal asks for the video path, enter the **absolute path** to the video you want to process. Do not put quotes around the path. For example:

```text
D:/path/to/your-video.mp4
```

A sample video will be available in the project root and can be used for a first run. You can also enter the absolute path to any other supported video.

The batch file also asks whether to display rejected detections. Enter `y` to show rejected detections as red boxes.

### Step 3: Wait for processing and view the results

Inference processes the video frame by frame, so wait for it to finish. Processing time depends on video length, resolution, and available hardware. The annotated video and CSV exports will be saved in `integration/outputs/`.

---

## 🖥️ System Requirements

- **OS:** Windows / Linux / macOS
- **Hardware:** A CUDA-enabled NVIDIA GPU is highly recommended for real-time video processing. It will run on a CPU, but processing long videos will be significantly slower.
- **Python:** 3.9+

## 📁 Where are the outputs?

Once the script finishes processing a video, the final `.mp4` output with drawn bounding boxes will be saved in the `integration/outputs/` directory.

For video input, inference also writes two CSV files to `integration/outputs/` while frames are processed:

- `<video>_detections.csv` is an audit file with each candidate that passed the D-FINE confidence threshold, including accepted/rejected status, decision reason, frame number, confidence, bounding-box width/height/area in pixels, and PIDNet ground and sky/vegetation ratios.
- `<video>_potholes_rds.csv` contains accepted detections only and uses the same 13 columns, in the same order, as `pothole_detections` in `schema.sql`. It can be loaded with PostgreSQL `COPY`/`\\copy`.

Rerunning inference for the same video preserves earlier outputs. The first run uses `<video>_verified.mp4`, `<video>_detections.csv`, and `<video>_potholes_rds.csv`; later runs use a numbered base such as `<video>_2_verified.mp4` and `<video>_2_potholes_rds.csv`. Each run uploads its own RDS CSV.

The pixel measurements are the dimensions/area of the detector's bounding rectangle, not a segmentation mask measurement. Device and vehicle IDs are random 128-bit integers represented as strings; each stays the same within one video run, and new IDs are generated on a later run. `detected_at` is an estimate: processing start time plus the frame's video offset. Each frame-level accepted detection receives a new UUID; temporal tracking is used for filtering but does not assign a stable pothole/event ID, so the same pothole may appear in multiple rows across frames.

Physical size cannot be reliably recovered from pixel dimensions alone: apparent size changes with distance and perspective, and depth is not observable from this single RGB camera pipeline. To keep the size fields populated, inference uses a rough fallback scale of 3 pixels/cm when no calibration is provided. It converts the larger and smaller dimensions of the detector bounding box into estimated length and width, and leaves `depth_cm` empty. Set `PIXELS_PER_CM` in `.env` (or pass `--pixels-per-cm`) to override the fallback with a scale calibrated for a known reference plane/distance. The audit CSV records the scale used. These are uncalibrated guesses by default, only meaningful near the reference plane, and should not be treated as measured dimensions.

Example import from `psql` after connecting to the RDS database (replace the local path):

```sql
\\copy pothole_detections (detection_id, device_id, vehicle_id, detected_at, latitude, longitude, length_cm, width_cm, depth_cm, severity, confidence, model_version, image_s3_key) FROM 'D:/.../integration/outputs/<video>_potholes_rds.csv' WITH (FORMAT csv, HEADER true, NULL '')
```

### Automatic AWS RDS upload

After each video finishes, the pipeline runs `integration/push_to_rds.py` on that video's accepted-detections CSV. Fill in the root `.env` file with the RDS endpoint, database name, username, and password. Keep the schema from `schema.sql` installed in the database and allow network access from the machine running inference. `RDS_SSLMODE=require` enables TLS; change it to `verify-full` if you configure a trusted CA certificate and hostname verification. The uploader skips the upload while required `.env` values are blank, and reports connection or database errors without interrupting completed inference.

You can also run the uploader manually:

```bash
python integration/push_to_rds.py integration/outputs/<video>_potholes_rds.csv
```

To upload the newest `*_potholes_rds.csv` in `integration/outputs/`, you can omit the CSV argument:

```bash
python integration/push_to_rds.py
```

The uploader inserts accepted frame-level rows and ignores duplicate `detection_id` values on retries.
