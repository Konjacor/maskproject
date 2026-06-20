from __future__ import annotations

from face_mask.core.types import ROI


class ROIManager:
    def build_upper_face_rois(self, width: int, height: int) -> dict[str, ROI]:
        face = ROI(int(width * 0.12), int(height * 0.1), int(width * 0.76), int(height * 0.45))
        rois = self.build_upper_face_rois_from_face(face)
        rois["face"] = face
        return rois

    def build_upper_face_rois_from_face(self, face: ROI) -> dict[str, ROI]:
        left_eye = ROI(
            x=face.x + int(face.width * 0.08),
            y=face.y + int(face.height * 0.40),
            width=int(face.width * 0.24),
            height=int(face.height * 0.20),
        )
        right_eye = ROI(
            x=face.x + int(face.width * 0.68),
            y=face.y + int(face.height * 0.40),
            width=int(face.width * 0.24),
            height=int(face.height * 0.20),
        )
        brow = ROI(
            x=face.x + int(face.width * 0.08),
            y=face.y + int(face.height * 0.18),
            width=int(face.width * 0.84),
            height=int(face.height * 0.16),
        )
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "brow": brow,
        }

    def build_full_face_rois(self, width: int, height: int) -> dict[str, ROI]:
        face = ROI(int(width * 0.12), int(height * 0.1), int(width * 0.76), int(height * 0.62))
        rois = self.build_upper_face_rois_from_face(face)
        rois["face"] = face
        rois["mouth"] = ROI(
            x=face.x + int(face.width * 0.28),
            y=face.y + int(face.height * 0.72),
            width=int(face.width * 0.44),
            height=int(face.height * 0.16),
        )
        return rois
