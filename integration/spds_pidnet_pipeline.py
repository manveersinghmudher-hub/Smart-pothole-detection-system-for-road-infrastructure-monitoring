"""
Smart Pothole Detection System (SPDS) - PIDNet + D-FINE Integrated Pipeline
=============================================================================
Combines PIDNet-S semantic road segmentation with D-FINE object detection to
filter out false-positive pothole detections that do not lie on drivable road surfaces.

Pipeline Architecture:
    Input Frame
        │
        ├──► PIDNet-S Road Segmentation ──► Road Mask
        │                                        │
        ├──► D-FINE Pothole Detection  ──► Raw Bounding Boxes
        │                                        │
        └───────────────────────────────► Verification Layer (road_ratio >= 0.40)
                                                 │
                                                 ▼
                                        Accepted Potholes
                                                 │
                                                 ▼
                                       Visualized Output & Profiling

Author: Senior Computer Vision Engineer
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SPDS_PIDNET_DFINE")

# ============================================================================
# DIRECTORY & DEFAULT PATHS
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DFINE_DIR = PROJECT_ROOT / "D-FINE-OFFICIAL"
PIDNET_DIR = PROJECT_ROOT / "PIDNet"

DEFAULT_DFINE_CONFIG = DFINE_DIR / "configs" / "dfine" / "dfine_hgnetv2_s_coco.yml"
DEFAULT_DFINE_WEIGHT = DFINE_DIR / "output" / "pothole_dfine_s_1000" / "best_stg1.pth"
FALLBACK_DFINE_WEIGHT = DFINE_DIR / "output" / "pothole_dfine_s_1000" / "checkpoint0923.pth"

DEFAULT_PIDNET_WEIGHT = (
    PIDNET_DIR / "pretrained_models" / "cityscapes" / "PIDNet_S_Cityscapes_test.pt"
)

DEFAULT_INPUT_DIR = SCRIPT_DIR / "inputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Supported Formats
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Cityscapes Class Groups for Spatial Verification
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole, 6: traffic light
# 7: traffic sign, 8: vegetation, 9: terrain, 10: sky, 11: person, 12: rider
# 13: car, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle
ROAD_CLASS_ID = 0
ROAD_CLASS_NAME = "road"
DRIVABLE_GROUND_CLASSES = {0, 1, 9}  # Road, Sidewalk/Shoulder, Terrain (gravel/dirt)
OFFROAD_OBSTACLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 10}  # Buildings, Walls, Fences, Poles, Signs, Trees/Foliage, Sky

# Normalization constants for PIDNet (ImageNet standard)
PIDNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIDNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# PERFORMANCE PROFILER MODULE
# ============================================================================
class PerformanceProfiler:
    """Tracks latency across pipeline stages and computes pipeline FPS."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = 0
        self.pidnet_time = 0.0
        self.dfine_time = 0.0
        self.verification_time = 0.0
        self.viz_time = 0.0
        self.total_time = 0.0

    def record(self, t_pidnet: float, t_dfine: float, t_verify: float, t_viz: float, t_total: float):
        self.counts += 1
        self.pidnet_time += t_pidnet
        self.dfine_time += t_dfine
        self.verification_time += t_verify
        self.viz_time += t_viz
        self.total_time += t_total

    def summary(self) -> Dict[str, Union[float, int]]:
        if self.counts == 0:
            return {}
        avg_pidnet = (self.pidnet_time / self.counts) * 1000.0
        avg_dfine = (self.dfine_time / self.counts) * 1000.0
        avg_verify = (self.verification_time / self.counts) * 1000.0
        avg_viz = (self.viz_time / self.counts) * 1000.0
        avg_total = (self.total_time / self.counts) * 1000.0
        fps = self.counts / self.total_time if self.total_time > 0 else 0.0

        return {
            "processed_frames": self.counts,
            "avg_pidnet_ms": avg_pidnet,
            "avg_dfine_ms": avg_dfine,
            "avg_verify_ms": avg_verify,
            "avg_viz_ms": avg_viz,
            "avg_total_ms": avg_total,
            "fps": fps,
        }

    def print_summary(self):
        stats = self.summary()
        if not stats:
            return
        logger.info("=" * 65)
        logger.info("PERFORMANCE PROFILING REPORT (%d frames)", stats["processed_frames"])
        logger.info("=" * 65)
        logger.info("  PIDNet Segmentation Latency : %6.2f ms (%4.1f%%)",
                    stats["avg_pidnet_ms"], (stats["avg_pidnet_ms"] / stats["avg_total_ms"]) * 100)
        logger.info("  D-FINE Detection Latency    : %6.2f ms (%4.1f%%)",
                    stats["avg_dfine_ms"], (stats["avg_dfine_ms"] / stats["avg_total_ms"]) * 100)
        logger.info("  Road Verification Latency   : %6.2f ms (%4.1f%%)",
                    stats["avg_verify_ms"], (stats["avg_verify_ms"] / stats["avg_total_ms"]) * 100)
        logger.info("  Visualization & Rendering   : %6.2f ms (%4.1f%%)",
                    stats["avg_viz_ms"], (stats["avg_viz_ms"] / stats["avg_total_ms"]) * 100)
        logger.info("  -------------------------------------------------------------")
        logger.info("  Total Pipeline Latency/Frame: %6.2f ms", stats["avg_total_ms"])
        logger.info("  Effective Pipeline FPS      : %6.2f FPS", stats["fps"])
        logger.info("=" * 65)


# ============================================================================
# PIDNET INTEGRATION MODULE
# ============================================================================
def load_pidnet(checkpoint_path: Union[str, Path], device: str) -> nn.Module:
    """
    Loads PIDNet-S with Cityscapes weights without downloading or retraining.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PIDNet checkpoint not found at: {checkpoint_path}")

    # Add PIDNet repository to path
    pidnet_root = str(PIDNET_DIR.resolve())
    if pidnet_root not in sys.path:
        sys.path.insert(0, pidnet_root)

    try:
        from models import pidnet
    except ImportError as exc:
        raise ImportError(f"Cannot import PIDNet models from {PIDNET_DIR}") from exc

    logger.info("Initializing PIDNet-S architecture (num_classes=19)...")
    model = pidnet.get_pred_model("pidnet-s", 19)

    logger.info("Loading PIDNet-S weights from: %s", checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    model_dict = model.state_dict()
    cleaned_state = {}
    for k, v in state.items():
        clean_key = k
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        elif clean_key.startswith("module."):
            clean_key = clean_key[7:]
        if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
            cleaned_state[clean_key] = v

    model_dict.update(cleaned_state)
    model.load_state_dict(model_dict, strict=False)

    model = model.to(device)
    model.eval()

    logger.info("PIDNet-S loaded successfully on device: %s", device)
    # Print Road Class ID requirement
    print(f"Road Class ID: {ROAD_CLASS_ID} ({ROAD_CLASS_NAME})")
    return model


def get_road_segmentation(
    model: nn.Module,
    image_bgr: np.ndarray,
    device: str,
    inference_size: Tuple[int, int] = (1024, 512)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes binary drivable ground mask and full 19-class Cityscapes segmentation map.

    Args:
        model: Evaluated PIDNet model.
        image_bgr: Input frame in BGR format.
        device: 'cuda' or 'cpu'.
        inference_size: (width, height) for PIDNet resolution.

    Returns:
        road_mask: Boolean numpy array (orig_h, orig_w) where True == drivable ground.
        seg_map: uint8 numpy array (orig_h, orig_w) containing 19 Cityscapes class IDs.
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # Resize to PIDNet-S inference resolution (1024 x 512)
    img_resized = cv2.resize(image_bgr, inference_size, interpolation=cv2.INTER_LINEAR)

    # Standard BGR -> RGB and ImageNet normalization
    img_norm = (img_resized.astype(np.float32)[:, :, ::-1] / 255.0 - PIDNET_MEAN) / PIDNET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)
        # Bilinear interpolation back to tensor resolution
        preds = F.interpolate(preds, size=tensor.size()[-2:], mode="bilinear", align_corners=True)
        pred_labels = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()

    # Nearest-neighbor resize prediction map back to original dimensions
    seg_map = cv2.resize(
        pred_labels.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Drivable ground mask: Road (0), Sidewalk (1), Terrain (9)
    road_mask = np.isin(seg_map, list(DRIVABLE_GROUND_CLASSES))
    return road_mask, seg_map


def get_road_mask(
    model: nn.Module,
    image_bgr: np.ndarray,
    device: str,
    inference_size: Tuple[int, int] = (1024, 512)
) -> np.ndarray:
    """Convenience wrapper returning binary ground mask."""
    road_mask, _ = get_road_segmentation(model, image_bgr, device, inference_size)
    return road_mask


# ============================================================================
# D-FINE INTEGRATION MODULE
# ============================================================================
class DFineDeployWrapper(nn.Module):
    """Wraps D-FINE core model and postprocessor in deploy mode."""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def load_dfine(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: str
) -> nn.Module:
    """Loads D-FINE pothole detector without retraining or modifying weights."""
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        if FALLBACK_DFINE_WEIGHT.exists():
            checkpoint_path = FALLBACK_DFINE_WEIGHT
        else:
            raise FileNotFoundError(f"D-FINE checkpoint not found at: {checkpoint_path}")

    dfine_root = str(DFINE_DIR.resolve())
    if dfine_root not in sys.path:
        sys.path.insert(0, dfine_root)

    try:
        from src.core import YAMLConfig
    except ImportError as exc:
        raise ImportError(f"Cannot import D-FINE core modules from {DFINE_DIR}") from exc

    logger.info("Initializing D-FINE from: %s", config_path)
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    cfg.yaml_cfg["num_classes"] = 1
    cfg.yaml_cfg["remap_mscoco_category"] = False

    logger.info("Loading D-FINE weights from: %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    if "ema" in checkpoint and "module" in checkpoint["ema"]:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cleaned_state = {}
    for k, v in state.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned_state[clean_key] = v

    cfg.model.load_state_dict(cleaned_state, strict=False)

    deploy_model = DFineDeployWrapper(cfg).to(device)
    deploy_model.eval()

    logger.info("D-FINE loaded successfully on device: %s", device)
    return deploy_model


def run_dfine(
    model: nn.Module,
    image_bgr: np.ndarray,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs D-FINE inference on input frame and returns raw boxes, scores, labels."""
    orig_h, orig_w = image_bgr.shape[:2]
    orig_size = torch.tensor([[orig_w, orig_h]], dtype=torch.float32, device=device)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])
    image_tensor = transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(image_tensor, orig_size)

    labels_np = labels[0].detach().cpu().numpy()
    boxes_np = boxes[0].detach().cpu().numpy()
    scores_np = scores[0].detach().cpu().numpy()

    return boxes_np, scores_np, labels_np


# ============================================================================
# VERIFICATION MODULE
# ============================================================================
class VerificationResult:
    """Encapsulates pothole verification data."""

    def __init__(
        self,
        box: np.ndarray,
        score: float,
        label: int,
        road_ratio: float,
        sky_veg_ratio: float,
        accepted: bool,
        reason: str = ""
    ):
        self.box = box
        self.score = score
        self.label = label
        self.road_ratio = road_ratio
        self.sky_veg_ratio = sky_veg_ratio
        self.accepted = accepted
        self.reason = reason


def verify_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    seg_map: np.ndarray,
    conf_threshold: float = 0.50,
    min_road_ratio: float = 0.20,
    max_box_area_ratio: float = 0.20,
    horizon_ratio: float = 0.30
) -> List[VerificationResult]:
    """
    Applies multi-criterion spatial verification tuned against real-world false positives:
      1. Max-area filter: Rejects giant boxes spanning > 20% of image area (sky/canopy artifacts).
      2. Horizon prior: Rejects boxes floating above the road horizon (y2 < top 30% of frame).
      3. Sky artifact exclusion: Rejects boxes with >= 20% Sky (Class 10).
      4. Off-road dense foliage filter: Rejects roadside bush detections (>= 50% Vegetation and < 10% Ground).
      5. Ground-plane acceptance: Accepts boxes with >= min_road_ratio (0.20) in Road (0), Sidewalk (1), or Terrain (9).
      6. Road-plane shadow safeguard: Overcomes sharp tree/pole shadows or dark pavement in the lower roadway
         (y2 > 45% of height, score >= 0.60, sky < 5%, veg < 35%).
    """
    orig_h, orig_w = seg_map.shape[:2]
    frame_area = orig_h * orig_w
    results: List[VerificationResult] = []

    valid_mask = scores >= conf_threshold
    indices = np.where(valid_mask)[0]

    for idx in indices:
        b = boxes[idx]
        score = float(scores[idx])
        label = int(labels[idx])

        # Clamp bounding box coordinates to image boundaries
        x1 = max(0, min(orig_w - 1, int(round(b[0]))))
        y1 = max(0, min(orig_h - 1, int(round(b[1]))))
        x2 = max(0, min(orig_w, int(round(b[2]))))
        y2 = max(0, min(orig_h, int(round(b[3]))))

        w_box = max(0, x2 - x1)
        h_box = max(0, y2 - y1)
        box_area = w_box * h_box
        area_ratio = (box_area / frame_area) if frame_area > 0 else 0.0

        # Criterion 1: Max-Area Filter (reject giant screen-filling hallucinations)
        if area_ratio > max_box_area_ratio:
            results.append(VerificationResult(
                box=b, score=score, label=label,
                road_ratio=0.0, sky_veg_ratio=0.0,
                accepted=False,
                reason=f"Area too large ({area_ratio*100:.1f}%)"
            ))
            continue

        # Criterion 2: Horizon Line Prior (reject floating objects in upper sky)
        if y2 < orig_h * horizon_ratio:
            results.append(VerificationResult(
                box=b, score=score, label=label,
                road_ratio=0.0, sky_veg_ratio=1.0,
                accepted=False,
                reason=f"Above horizon (y2={y2})"
            ))
            continue

        # Crop corresponding region from 19-class segmentation map
        box_labels = seg_map[y1:y2, x1:x2]
        total_pixels = box_labels.size

        if total_pixels == 0:
            results.append(VerificationResult(
                box=b, score=score, label=label,
                road_ratio=0.0, sky_veg_ratio=0.0,
                accepted=False,
                reason="Empty crop"
            ))
            continue

        # Class 10: Sky
        sky_pixels = int(np.count_nonzero(box_labels == 10))
        sky_ratio = sky_pixels / total_pixels

        # Class 8: Vegetation / Foliage
        veg_pixels = int(np.count_nonzero(box_labels == 8))
        veg_ratio = veg_pixels / total_pixels

        # Drivable ground: Road (0), Sidewalk/Shoulder (1), Terrain (9)
        ground_pixels = int(np.count_nonzero(np.isin(box_labels, list(DRIVABLE_GROUND_CLASSES))))
        ground_ratio = ground_pixels / total_pixels

        # Criterion 3: Predominantly Sky -> REJECT
        if sky_ratio >= 0.20:
            accepted = False
            reason = f"Sky artifact ({sky_ratio*100:.1f}%)"

        # Criterion 4: Dense Off-Road Foliage -> REJECT
        elif veg_ratio >= 0.50 and ground_ratio < 0.10:
            accepted = False
            reason = f"Off-road foliage ({veg_ratio*100:.1f}%)"

        # Criterion 5: Drivable ground surface -> ACCEPT
        elif ground_ratio >= min_road_ratio:
            accepted = True
            reason = f"Ground ({ground_ratio*100:.1f}%)"

        # Criterion 6: Road-Plane & Shadow Safeguard
        # Overcomes sharp tree/pole shadows (misclassified by Cityscapes as Class 2 Building)
        # or dark pavement texture in the forward roadway plane
        elif score >= 0.60 and y2 > orig_h * 0.45 and sky_ratio < 0.05 and veg_ratio < 0.35:
            accepted = True
            reason = f"Road-plane shadow override (conf={score:.2f})"

        else:
            accepted = False
            reason = f"Low ground ({ground_ratio*100:.1f}%)"

        results.append(VerificationResult(
            box=b,
            score=score,
            label=label,
            road_ratio=ground_ratio,
            sky_veg_ratio=sky_ratio + veg_ratio,
            accepted=accepted,
            reason=reason
        ))

    return results


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return (intersection / union) if union > 0 else 0.0


class TemporalPotholeTracker:
    """Tracks potholes across video frames to eliminate shadow-induced flicker and ensure continuity."""

    def __init__(self, max_age: int = 15, iou_thresh: float = 0.20):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks: List[Dict[str, Any]] = []

    def update(
        self,
        results: List[VerificationResult],
        orig_w: int,
        orig_h: int
    ) -> List[VerificationResult]:
        matched_track_indices = set()

        # Phase 1: Match accepted detections to existing tracks or create new tracks
        for r in results:
            if r.accepted:
                best_iou = 0.0
                best_idx = -1
                for idx, tr in enumerate(self.tracks):
                    if idx in matched_track_indices:
                        continue
                    iou = compute_iou(r.box, tr['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_idx >= 0 and best_iou >= self.iou_thresh:
                    self.tracks[best_idx]['box'] = 0.6 * r.box + 0.4 * self.tracks[best_idx]['box']
                    self.tracks[best_idx]['age'] = 0
                    self.tracks[best_idx]['hits'] += 1
                    matched_track_indices.add(best_idx)
                else:
                    self.tracks.append({'box': r.box.copy(), 'age': 0, 'hits': 1})
                    matched_track_indices.add(len(self.tracks) - 1)

        # Phase 2: For rejected detections, check if they match an established track (hits >= 2)
        for r in results:
            if not r.accepted:
                if r.box[3] > orig_h * 0.40 and r.sky_veg_ratio < 0.20:
                    best_iou = 0.0
                    best_idx = -1
                    for idx, tr in enumerate(self.tracks):
                        if tr['hits'] >= 2:
                            iou = compute_iou(r.box, tr['box'])
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = idx

                    if best_idx >= 0 and best_iou >= self.iou_thresh:
                        r.accepted = True
                        r.reason = "Tracked persistence"
                        self.tracks[best_idx]['age'] = 0
                        self.tracks[best_idx]['hits'] += 1
                        matched_track_indices.add(best_idx)

        # Age out unmatched tracks
        surviving = []
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_track_indices:
                tr['age'] += 1
            if tr['age'] <= self.max_age:
                surviving.append(tr)
        self.tracks = surviving

        return results


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================
def draw_pipeline_results(
    image_bgr: np.ndarray,
    verified_results: List[VerificationResult],
    show_rejected: bool = False,
    blend_road_overlay: bool = False,
    road_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Renders accepted pothole boxes, confidence scores, and road verification tags.
    Optionally displays rejected boxes and a translucent road segmentation overlay.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # Optional subtle green road tint
    if blend_road_overlay and road_mask is not None:
        overlay = annotated.copy()
        overlay[road_mask] = (
            overlay[road_mask] * 0.75 + np.array([0, 200, 0], dtype=np.uint8) * 0.25
        ).astype(np.uint8)
        annotated = overlay

    for item in verified_results:
        if not item.accepted and not show_rejected:
            continue

        b = item.box
        x1 = max(0, min(w - 1, int(round(b[0]))))
        y1 = max(0, min(h - 1, int(round(b[1]))))
        x2 = max(0, min(w - 1, int(round(b[2]))))
        y2 = max(0, min(h - 1, int(round(b[3]))))

        if item.accepted:
            box_color = (0, 255, 0)      # Bright Green
            tag_color = (0, 180, 0)
            ground_tag = f"ground {item.road_ratio:.0%}" if item.road_ratio > 0 else "road plane"
            status_text = f"pothole {item.score:.2f} ({ground_tag})"
        else:
            box_color = (0, 0, 255)      # Red for rejected
            tag_color = (0, 0, 180)
            reason_str = f" - {item.reason}" if item.reason else ""
            status_text = f"REJECTED {item.score:.2f}{reason_str}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Label Banner - ensure it never clips off the top or right of the screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)

        if y1 - th - baseline - 6 < 0:
            # Box is at the very top of image; draw banner inside top of box
            label_y1 = y1
            label_y2 = min(h - 1, y1 + th + baseline + 6)
            text_y = y1 + th + 2
        else:
            # Draw banner above the box
            label_y1 = max(0, y1 - th - baseline - 6)
            label_y2 = y1
            text_y = y1 - 4

        label_x2 = min(w - 1, x1 + tw + 8)

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), tag_color, cv2.FILLED)
        cv2.putText(
            annotated,
            status_text,
            (x1 + 4, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated


def save_validation_road_mask(road_mask: np.ndarray, output_path: Path):
    """Saves a binary road mask image for road class validation."""
    mask_visual = (road_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), mask_visual)
    logger.info("Saved sample road mask for validation: %s", output_path)


# ============================================================================
# IMAGE INFERENCE
# ============================================================================
def process_image(
    pidnet_model: nn.Module,
    dfine_model: nn.Module,
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str,
    conf_threshold: float = 0.50,
    min_road_ratio: float = 0.25,
    show_rejected: bool = False,
    profiler: Optional[PerformanceProfiler] = None
) -> Path:
    """Performs integrated PIDNet + D-FINE verification on a single image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    # Stage 1: PIDNet Ground Segmentation & 19-class Map
    t0 = time.perf_counter()
    road_mask, seg_map = get_road_segmentation(pidnet_model, image_bgr, device)
    t_pidnet = time.perf_counter() - t0

    # Save road mask sample for validation
    val_mask_path = output_dir / "road_mask_validation.png"
    if not val_mask_path.exists():
        save_validation_road_mask(road_mask, val_mask_path)

    # Stage 2: D-FINE Detection
    t0 = time.perf_counter()
    boxes, scores, labels = run_dfine(dfine_model, image_bgr, device)
    t_dfine = time.perf_counter() - t0

    # Stage 3: Verification Layer
    t0 = time.perf_counter()
    results = verify_detections(
        boxes, scores, labels, seg_map,
        conf_threshold=conf_threshold,
        min_road_ratio=min_road_ratio
    )
    t_verify = time.perf_counter() - t0

    accepted_count = sum(1 for r in results if r.accepted)
    rejected_count = len(results) - accepted_count

    # Stage 4: Visualization
    t0 = time.perf_counter()
    annotated = draw_pipeline_results(
        image_bgr, results,
        show_rejected=show_rejected,
        blend_road_overlay=False,
        road_mask=road_mask
    )
    output_path = output_dir / f"{image_path.stem}_verified{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)
    t_viz = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    if profiler is not None:
        profiler.record(t_pidnet, t_dfine, t_verify, t_viz, t_total)

    logger.info("Image: %s | D-FINE Raw: %d | Accepted: %d | Rejected: %d",
                image_path.name, len(results), accepted_count, rejected_count)
    logger.info("Saved annotated image to: %s", output_path)

    return output_path


# ============================================================================
# VIDEO INFERENCE
# ============================================================================
def process_video(
    pidnet_model: nn.Module,
    dfine_model: nn.Module,
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str,
    conf_threshold: float = 0.50,
    min_road_ratio: float = 0.25,
    show_rejected: bool = False,
    profiler: Optional[PerformanceProfiler] = None
) -> Path:
    """Performs integrated PIDNet + D-FINE verification frame-by-frame on video."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / f"{video_path.stem}_verified.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    logger.info("Processing video: %s (%d frames, %dx%d @ %.1f FPS)",
                video_path.name, total_frames, w, h, fps)

    frame_idx = 0
    total_accepted = 0
    total_rejected = 0
    validation_mask_saved = False
    tracker = TemporalPotholeTracker(max_age=15, iou_thresh=0.20)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_idx += 1
            t_start = time.perf_counter()

            # 1. PIDNet ground segmentation & 19-class map
            t0 = time.perf_counter()
            road_mask, seg_map = get_road_segmentation(pidnet_model, frame_bgr, device)
            t_pidnet = time.perf_counter() - t0

            # Save first frame road mask for validation
            if not validation_mask_saved:
                val_mask_path = output_dir / "road_mask_validation.png"
                save_validation_road_mask(road_mask, val_mask_path)
                validation_mask_saved = True

            # 2. D-FINE inference
            t0 = time.perf_counter()
            boxes, scores, labels = run_dfine(dfine_model, frame_bgr, device)
            t_dfine = time.perf_counter() - t0

            # 3. Verification & Temporal Tracking
            t0 = time.perf_counter()
            results = verify_detections(
                boxes, scores, labels, seg_map,
                conf_threshold=conf_threshold,
                min_road_ratio=min_road_ratio
            )
            results = tracker.update(results, w, h)
            t_verify = time.perf_counter() - t0

            accepted_in_frame = sum(1 for r in results if r.accepted)
            rejected_in_frame = len(results) - accepted_in_frame
            total_accepted += accepted_in_frame
            total_rejected += rejected_in_frame

            # 4. Visualization & write
            t0 = time.perf_counter()
            annotated_frame = draw_pipeline_results(
                frame_bgr, results,
                show_rejected=show_rejected,
                blend_road_overlay=False,
                road_mask=road_mask
            )
            out.write(annotated_frame)
            t_viz = time.perf_counter() - t0

            t_total = time.perf_counter() - t_start
            if profiler is not None:
                profiler.record(t_pidnet, t_dfine, t_verify, t_viz, t_total)

            if frame_idx == 1 or frame_idx % 30 == 0 or frame_idx == total_frames:
                frame_info = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
                logger.info("Frame %s | Accepted: %d | Rejected: %d",
                            frame_info, accepted_in_frame, rejected_in_frame)

    finally:
        cap.release()
        out.release()

    logger.info("Finished video processing. Total frames: %d | Total Accepted: %d | Total Rejected: %d",
                frame_idx, total_accepted, total_rejected)
    logger.info("Output video saved to: %s", output_path)
    return output_path


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPDS - PIDNet Road Segmentation + D-FINE Pothole Detection Pipeline"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Input image, video, or folder path (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--pidnet-weight",
        type=str,
        default=str(DEFAULT_PIDNET_WEIGHT),
        help=f"Path to PIDNet checkpoint (default: {DEFAULT_PIDNET_WEIGHT})"
    )
    parser.add_argument(
        "--dfine-config",
        type=str,
        default=str(DEFAULT_DFINE_CONFIG),
        help=f"Path to D-FINE YAML config (default: {DEFAULT_DFINE_CONFIG})"
    )
    parser.add_argument(
        "--dfine-weight",
        type=str,
        default=str(DEFAULT_DFINE_WEIGHT),
        help=f"Path to D-FINE checkpoint (default: {DEFAULT_DFINE_WEIGHT})"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.50,
        help="Confidence threshold for D-FINE (default: 0.50)"
    )
    parser.add_argument(
        "--min-road-ratio",
        type=float,
        default=0.20,
        help="Minimum road surface pixel ratio to accept pothole (default: 0.20)"
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Display rejected off-road detections in red for debugging"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Compute device: 'cuda' or 'cpu' (default: auto-detect)"
    )

    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("SPDS: PIDNet Road Segmentation + D-FINE Pothole Detection")
    logger.info("=" * 65)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1. Load PIDNet
    pidnet_model = load_pidnet(args.pidnet_weight, device)

    # 2. Load D-FINE
    dfine_model = load_dfine(args.dfine_config, args.dfine_weight, device)

    # Discover inputs
    input_target = Path(args.input)
    images: List[Path] = []
    videos: List[Path] = []

    if input_target.is_file():
        suffix = input_target.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTS:
            images.append(input_target)
        elif suffix in SUPPORTED_VIDEO_EXTS:
            videos.append(input_target)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    elif input_target.is_dir():
        for f in sorted(input_target.iterdir()):
            if f.is_file():
                s = f.suffix.lower()
                if s in SUPPORTED_IMAGE_EXTS:
                    images.append(f)
                elif s in SUPPORTED_VIDEO_EXTS:
                    videos.append(f)
    else:
        raise FileNotFoundError(f"Input path not found: {input_target}")

    total_items = len(images) + len(videos)
    if total_items == 0:
        logger.warning("No supported files found in: %s", input_target)
        return

    logger.info("Processing %d item(s): %d image(s), %d video(s)",
                total_items, len(images), len(videos))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = PerformanceProfiler()

    for img_path in images:
        process_image(
            pidnet_model=pidnet_model,
            dfine_model=dfine_model,
            image_path=img_path,
            output_dir=output_dir,
            device=device,
            conf_threshold=args.threshold,
            min_road_ratio=args.min_road_ratio,
            show_rejected=args.show_rejected,
            profiler=profiler
        )

    for vid_path in videos:
        process_video(
            pidnet_model=pidnet_model,
            dfine_model=dfine_model,
            video_path=vid_path,
            output_dir=output_dir,
            device=device,
            conf_threshold=args.threshold,
            min_road_ratio=args.min_road_ratio,
            show_rejected=args.show_rejected,
            profiler=profiler
        )

    # Print profiling table
    profiler.print_summary()
    logger.info("All processing completed. Results in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
