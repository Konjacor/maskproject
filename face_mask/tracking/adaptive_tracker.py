from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from face_mask.core.types import FramePacket, ROI, TrackingResult
from face_mask.features.lighting_feature import compute_lighting_score
from face_mask.features.visibility_feature import compute_visibility_score
from face_mask.tracking.base import BaseTracker
from face_mask.tracking.roi_manager import ROIManager
from face_mask.utils.image_ops import extract_roi


@dataclass(slots=True)
class AdaptiveTrackerState:
    face_roi: ROI | None = None
    confidence: float = 0.0
    lost_frames: int = 0


class AdaptiveVisibleRegionTracker(BaseTracker):
    def __init__(self, tracking_mode: str = "adaptive_visible_region", config: dict[str, Any] | None = None) -> None:
        self.tracking_mode = tracking_mode
        self.config = config or {}
        self.roi_manager = ROIManager()
        self.state = AdaptiveTrackerState()
        tracking_config = self.config.get("tracking", {})
        self.search_margin = float(tracking_config.get("search_margin", 0.22))
        self.relock_frames = int(tracking_config.get("relock_frames", 5))
        self.min_tracking_confidence = float(tracking_config.get("min_tracking_confidence", 0.28))

    def track(self, frame: FramePacket) -> TrackingResult:
        gray = frame.gray if frame.gray is not None else frame.image
        height, width = gray.shape[:2]
        visibility_score = compute_visibility_score(gray)
        lighting_score = compute_lighting_score(gray)

        coarse_face = self._locate_face(gray)
        notes: list[str] = []

        if self._should_relock():
            face_roi = coarse_face
            notes.append("relock")
        else:
            local_face = self._refine_face(gray, self.state.face_roi or coarse_face)
            coarse_score = self._score_roi(gray, coarse_face)
            local_score = self._score_roi(gray, local_face)
            if local_score >= coarse_score * 0.82:
                face_roi = local_face
                notes.append("local_follow")
            else:
                face_roi = coarse_face
                notes.append("global_refresh")

        face_roi = face_roi.clamp(width, height)
        rois = self.roi_manager.build_upper_face_rois_from_face(face_roi).copy()
        rois["face"] = face_roi

        tracking_confidence = min(1.0, max(0.0, self._score_roi(gray, face_roi)))
        if tracking_confidence < self.min_tracking_confidence:
            notes.append("low_tracking_confidence")
            self.state.lost_frames += 1
        else:
            self.state.face_roi = face_roi
            self.state.confidence = tracking_confidence
            self.state.lost_frames = 0

        if visibility_score < 0.35:
            notes.append("low_visibility")
        if lighting_score < 0.3:
            notes.append("lighting_instability")

        return TrackingResult(
            tracking_mode=self.tracking_mode,
            visible_regions=rois,
            visibility_score=min(1.0, max(0.0, visibility_score * 0.6 + tracking_confidence * 0.4)),
            face_present=tracking_confidence > 0.12,
            lighting_score=lighting_score,
            notes=notes,
        )

    def _should_relock(self) -> bool:
        return self.state.face_roi is None or self.state.lost_frames >= self.relock_frames

    def _locate_face(self, gray: np.ndarray) -> ROI:
        height, width = gray.shape[:2]
        template = self.roi_manager.build_upper_face_rois(width, height)["face"]
        search_y0 = int(height * 0.05)
        search_y1 = int(height * 0.72)
        window_w = template.width
        window_h = template.height
        stride_x = max(12, window_w // 8)
        stride_y = max(12, window_h // 8)

        best_roi = template
        best_score = -1.0
        for y in range(search_y0, max(search_y0 + 1, search_y1 - window_h), stride_y):
            for x in range(0, max(1, width - window_w), stride_x):
                roi = ROI(x=x, y=y, width=window_w, height=window_h).clamp(width, height)
                score = self._score_roi(gray, roi)
                if score > best_score:
                    best_score = score
                    best_roi = roi
        return best_roi

    def _refine_face(self, gray: np.ndarray, base_roi: ROI) -> ROI:
        height, width = gray.shape[:2]
        search_roi = base_roi.scale_from_center(1.0 + self.search_margin, 1.0 + self.search_margin).clamp(width, height)
        shift_x = max(6, int(base_roi.width * 0.08))
        shift_y = max(6, int(base_roi.height * 0.08))
        size_delta_w = max(4, int(base_roi.width * 0.06))
        size_delta_h = max(4, int(base_roi.height * 0.06))

        best_roi = base_roi.clamp(width, height)
        best_score = self._score_roi(gray, best_roi)
        for dx in (-shift_x, 0, shift_x):
            for dy in (-shift_y, 0, shift_y):
                for dw in (-size_delta_w, 0, size_delta_w):
                    for dh in (-size_delta_h, 0, size_delta_h):
                        candidate = ROI(
                            x=base_roi.x + dx,
                            y=base_roi.y + dy,
                            width=max(10, base_roi.width + dw),
                            height=max(10, base_roi.height + dh),
                        ).clamp(width, height)
                        if not self._inside(candidate, search_roi):
                            continue
                        score = self._score_roi(gray, candidate)
                        if score > best_score:
                            best_score = score
                            best_roi = candidate
        return best_roi

    def _score_roi(self, gray: np.ndarray, roi: ROI) -> float:
        region = extract_roi(gray, roi.x, roi.y, roi.width, roi.height)
        if region.size == 0:
            return 0.0
        normalized = region.astype(np.float32) / 255.0
        mean = float(np.mean(normalized))
        std = float(np.std(normalized))
        edges = cv2.Canny(region, 40, 120)
        edge_density = float(np.mean(edges > 0))
        return max(0.0, min(1.0, std * 1.8 + edge_density * 1.4 + (1.0 - abs(mean - 0.45))))

    def _inside(self, candidate: ROI, search_roi: ROI) -> bool:
        return (
            candidate.x >= search_roi.x
            and candidate.y >= search_roi.y
            and candidate.x + candidate.width <= search_roi.x + search_roi.width
            and candidate.y + candidate.height <= search_roi.y + search_roi.height
        )
