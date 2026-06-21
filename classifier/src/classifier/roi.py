from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_mask.core.types import FramePacket, ROI
from face_mask.tracking.detector_tracker import DetectorTracker
from face_mask.tracking.upper_face_tracker import UpperFaceTracker
from face_mask.utils.image_ops import normalize_local_contrast, to_grayscale


def build_tracker(config: dict[str, Any]):
    tracking_config = {
        "tracking": {
            "face_cascade_path": config.get("tracking", {}).get("face_cascade_path", ""),
            "min_face_area_ratio": config.get("tracking", {}).get("min_face_area_ratio", 0.06),
            "relock_frames": config.get("tracking", {}).get("relock_frames", 6),
            "local_margin": config.get("tracking", {}).get("local_margin", 0.18),
        }
    }
    backend = config.get("tracking", {}).get("backend", "detector")
    mode = config.get("tracking", {}).get("mode", "upper_face")
    if backend == "detector":
        return DetectorTracker(tracking_mode=mode, config=tracking_config)
    return UpperFaceTracker(tracking_mode=mode)


def open_camera(config: dict[str, Any]) -> cv2.VideoCapture:
    camera_config = config.get("camera", {})
    capture = cv2.VideoCapture(camera_config.get("index", 0))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config.get("width", 640)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config.get("height", 480)))
    capture.set(cv2.CAP_PROP_FPS, int(camera_config.get("fps", 20)))
    return capture


def read_packet(capture: cv2.VideoCapture, frame_id: int, config: dict[str, Any]) -> FramePacket:
    ok, image = capture.read()
    if not ok:
        raise RuntimeError("Failed to read camera frame")
    camera_config = config.get("camera", {})
    if camera_config.get("flip_horizontal", False):
        image = cv2.flip(image, 1)
    if camera_config.get("flip_vertical", False):
        image = cv2.flip(image, 0)
    gray = to_grayscale(image)
    if camera_config.get("normalize_contrast", True):
        gray = normalize_local_contrast(gray)
    return FramePacket(frame_id=frame_id, timestamp_monotonic=0.0, image=image, gray=gray, metadata={})


def extract_upper_face_crop(frame: FramePacket, face_roi: ROI, config: dict[str, Any]) -> np.ndarray:
    margin = float(config.get("capture", {}).get("roi_margin", 0.08))
    expanded = face_roi.scale_from_center(1.0 + margin * 2, 1.0 + margin * 1.5).clamp(frame.image.shape[1], frame.image.shape[0])
    crop = frame.gray[expanded.y : expanded.y + expanded.height, expanded.x : expanded.x + expanded.width]
    input_size = int(config.get("input_size", 96))
    resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
    if not config.get("grayscale", True):
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized


def save_debug_frame(path: str | Path, frame: FramePacket, tracking_result, label: str) -> None:
    image = frame.image.copy()
    for name, roi in tracking_result.visible_regions.items():
        cv2.rectangle(image, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (0, 255, 0), 2)
        cv2.putText(image, name, (roi.x, max(16, roi.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"label={label}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
