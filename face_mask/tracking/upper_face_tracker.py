from __future__ import annotations

from face_mask.core.types import FramePacket, TrackingResult
from face_mask.features.lighting_feature import compute_lighting_score
from face_mask.features.visibility_feature import compute_visibility_score
from face_mask.tracking.base import BaseTracker
from face_mask.tracking.roi_manager import ROIManager


class UpperFaceTracker(BaseTracker):
    def __init__(self, tracking_mode: str = "upper_face") -> None:
        self.tracking_mode = tracking_mode
        self.roi_manager = ROIManager()

    def track(self, frame: FramePacket) -> TrackingResult:
        height, width = frame.image.shape[:2]
        gray = frame.gray if frame.gray is not None else frame.image
        rois = self.roi_manager.build_upper_face_rois(width, height)
        visibility_score = compute_visibility_score(gray)
        lighting_score = compute_lighting_score(gray)
        notes: list[str] = []
        if visibility_score < 0.35:
            notes.append("low_visibility")
        if lighting_score < 0.3:
            notes.append("lighting_instability")
        return TrackingResult(
            tracking_mode=self.tracking_mode,
            visible_regions=rois,
            visibility_score=visibility_score,
            face_present=visibility_score > 0.1,
            lighting_score=lighting_score,
            notes=notes,
        )
