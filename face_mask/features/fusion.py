from __future__ import annotations

from face_mask.core.types import FeatureBundle, FramePacket, TrackingResult
from face_mask.features.brow_feature import compute_brow_raise_score
from face_mask.features.eye_feature import compute_eye_open_score
from face_mask.utils.image_ops import extract_roi


def build_feature_bundle(frame: FramePacket, tracking: TrackingResult) -> FeatureBundle:
    gray = frame.gray if frame.gray is not None else frame.image
    regions = tracking.visible_regions

    left_eye_roi = extract_roi(gray, **regions["left_eye"].__dict__)
    right_eye_roi = extract_roi(gray, **regions["right_eye"].__dict__)
    brow_roi = extract_roi(gray, **regions["brow"].__dict__)

    left_eye = compute_eye_open_score(left_eye_roi)
    right_eye = compute_eye_open_score(right_eye_roi)
    eye_open = (left_eye + right_eye) / 2.0
    brow_raise = compute_brow_raise_score(brow_roi, eye_roi=(left_eye_roi + right_eye_roi) / 2 if left_eye_roi.shape == right_eye_roi.shape else brow_roi)

    confidence = min(1.0, max(0.0, tracking.visibility_score * 0.6 + tracking.lighting_score * 0.4))

    return FeatureBundle(
        eye_open=eye_open,
        brow_raise=brow_raise,
        visibility_score=tracking.visibility_score,
        lighting_score=tracking.lighting_score,
        confidence=confidence,
        raw={
            "left_eye_open": left_eye,
            "right_eye_open": right_eye,
            "brow_raise": brow_raise,
        },
    )
