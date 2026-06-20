from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_mask.core.types import FramePacket, ROI, TrackingResult
from face_mask.features.lighting_feature import compute_lighting_score
from face_mask.features.visibility_feature import compute_visibility_score
from face_mask.tracking.base import BaseTracker
from face_mask.tracking.roi_manager import ROIManager


@dataclass(slots=True)
class DetectorTrackerState:
    face_roi: ROI | None = None
    lost_frames: int = 0
    confidence: float = 0.0


class DetectorTracker(BaseTracker):
    def __init__(self, tracking_mode: str = "detector", config: dict[str, Any] | None = None) -> None:
        self.tracking_mode = tracking_mode
        self.config = config or {}
        self.roi_manager = ROIManager()
        self.state = DetectorTrackerState()
        tracking_config = self.config.get("tracking", {})
        configured_cascade_path = tracking_config.get("face_cascade_path", "")
        self.face_cascade_path = configured_cascade_path or str(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self.min_face_area_ratio = float(tracking_config.get("min_face_area_ratio", 0.06))
        self.relock_frames = int(tracking_config.get("relock_frames", 6))
        self.local_margin = float(tracking_config.get("local_margin", 0.18))
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {self.face_cascade_path}")

    def track(self, frame: FramePacket) -> TrackingResult:
        gray = frame.gray if frame.gray is not None else frame.image
        height, width = gray.shape[:2]
        visibility_score = compute_visibility_score(gray)
        lighting_score = compute_lighting_score(gray)
        notes: list[str] = []

        face_roi, detection_notes = self._detect_face(gray)
        notes.extend(detection_notes)
        face_roi = face_roi.clamp(width, height)

        if self.state.face_roi is not None and self.state.lost_frames < self.relock_frames:
            face_roi = self._blend_rois(self.state.face_roi, face_roi, width, height)
            notes.append("tracked_blend")

        tracking_confidence = self._score_face(gray, face_roi)
        if tracking_confidence < 0.18:
            self.state.lost_frames += 1
            notes.append("low_tracking_confidence")
        else:
            self.state.face_roi = face_roi
            self.state.lost_frames = 0
            self.state.confidence = tracking_confidence

        rois = self.roi_manager.build_upper_face_rois_from_face(face_roi).copy()
        rois["face"] = face_roi

        if visibility_score < 0.35:
            notes.append("low_visibility")
        if lighting_score < 0.3:
            notes.append("lighting_instability")

        return TrackingResult(
            tracking_mode=self.tracking_mode,
            visible_regions=rois,
            visibility_score=min(1.0, max(0.0, visibility_score * 0.5 + tracking_confidence * 0.5)),
            face_present=tracking_confidence > 0.12,
            lighting_score=lighting_score,
            notes=notes,
        )

    def _detect_face(self, gray: np.ndarray) -> tuple[ROI, list[str]]:
        height, width = gray.shape[:2]
        min_size = (
            max(40, int(width * self.min_face_area_ratio)),
            max(40, int(height * self.min_face_area_ratio)),
        )
        candidates = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=min_size,
        )
        notes: list[str] = []
        if len(candidates) == 0:
            notes.append("detector_fallback")
            return self._fallback_face(gray), notes

        best = max(candidates, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = map(int, best)
        detected = ROI(x=x, y=y, width=w, height=h)
        notes.append("detector_face")
        return detected, notes

    def _fallback_face(self, gray: np.ndarray) -> ROI:
        height, width = gray.shape[:2]
        return self.roi_manager.build_upper_face_rois(width, height)["face"]

    def _blend_rois(self, previous: ROI, current: ROI, width: int, height: int) -> ROI:
        alpha = 0.7
        x = int(previous.x * alpha + current.x * (1.0 - alpha))
        y = int(previous.y * alpha + current.y * (1.0 - alpha))
        w = int(previous.width * alpha + current.width * (1.0 - alpha))
        h = int(previous.height * alpha + current.height * (1.0 - alpha))
        expanded = ROI(x=x, y=y, width=max(20, w), height=max(20, h))
        return expanded.clamp(width, height)

    def _score_face(self, gray: np.ndarray, roi: ROI) -> float:
        region = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        if region.size == 0:
            return 0.0
        normalized = region.astype(np.float32) / 255.0
        std = float(np.std(normalized))
        edges = cv2.Canny(region, 40, 120)
        edge_density = float(np.mean(edges > 0))
        area_ratio = (roi.width * roi.height) / float(gray.shape[0] * gray.shape[1])
        area_score = 1.0 - abs(area_ratio - 0.18) / 0.18
        return max(0.0, min(1.0, std * 1.2 + edge_density * 1.0 + max(0.0, area_score)))
