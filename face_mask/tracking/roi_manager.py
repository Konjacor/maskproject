from __future__ import annotations

from face_mask.core.types import ROI


class ROIManager:
    def build_upper_face_rois(self, width: int, height: int) -> dict[str, ROI]:
        face = ROI(int(width * 0.12), int(height * 0.1), int(width * 0.76), int(height * 0.45))
        left_eye = ROI(int(width * 0.18), int(height * 0.28), int(width * 0.22), int(height * 0.12))
        right_eye = ROI(int(width * 0.60), int(height * 0.28), int(width * 0.22), int(height * 0.12))
        brow = ROI(int(width * 0.18), int(height * 0.16), int(width * 0.64), int(height * 0.10))
        return {
            "face": face,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "brow": brow,
        }

    def build_full_face_rois(self, width: int, height: int) -> dict[str, ROI]:
        rois = self.build_upper_face_rois(width, height)
        rois["mouth"] = ROI(int(width * 0.30), int(height * 0.56), int(width * 0.40), int(height * 0.16))
        return rois
