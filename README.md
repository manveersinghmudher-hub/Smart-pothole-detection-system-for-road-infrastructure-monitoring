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

### Step 1: Install Dependencies

Open your terminal and install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 2: Download Model Weights

Due to file size limits, the pre-trained weights are hosted externally.

1. Download the D-FINE weights here: https://drive.google.com/file/d/1KAl_I7tOBeoUfz29PxHCNg64TRso5QWI/view?usp=drive_link

2. Download the PIDNet weights here: https://drive.google.com/file/d/17NF9dZj44dLT_8ZNJ6TeU0niMdgB8I2h/view?usp=drive_link

3. Place them in the following folder structure:
   - `D-FINE-OFFICIAL/output/pothole_dfine_s_1000/best_stg1.pth`
   - `PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt`

### Step 3: Run the Inference!

**On Windows:**
Simply double-click the **`run_inference.bat`** file. A terminal will open and ask you for the path to your video.

**On Mac/Linux (or via Terminal):**

```bash
python integration/spds_pidnet_pipeline.py -i "path/to/your/video.mp4"
```

_(Tip: Pass `--show-rejected` in the terminal or type `y` in the batch script to see the false-positives that our PIDNet layer successfully caught and filtered out, drawn as red boxes!)_

---

## 🖥️ System Requirements

- **OS:** Windows / Linux / macOS
- **Hardware:** A CUDA-enabled NVIDIA GPU is highly recommended for real-time video processing. It will run on a CPU, but processing long videos will be significantly slower.
- **Python:** 3.9+

## 📁 Where are the outputs?

Once the script finishes processing a video, the final `.mp4` output with drawn bounding boxes will be saved in the `integration/outputs/` directory.
