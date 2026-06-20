from __future__ import annotations

import time
from typing import Any

import cv2

from face_mask.core.types import FramePacket
from face_mask.sensors.base import BaseSensor
from face_mask.utils.image_ops import normalize_local_contrast, to_grayscale


class CameraSensor(BaseSensor):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.frame_id = 0
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        camera_index = self.config.get("camera_index", 0)
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("width", 640)))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("height", 480)))
        self.capture.set(cv2.CAP_PROP_FPS, int(self.config.get("fps", 20)))

    def read(self) -> FramePacket:
        if self.capture is None:
            raise RuntimeError("Camera sensor is not open")
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame")

        if self.config.get("flip_horizontal", False):
            image = cv2.flip(image, 1)
        if self.config.get("flip_vertical", False):
            image = cv2.flip(image, 0)

        gray = to_grayscale(image)
        if self.config.get("normalize_contrast", True):
            gray = normalize_local_contrast(gray)

        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp_monotonic=time.monotonic(),
            image=image,
            gray=gray,
            metadata={
                "mock": False,
                "infrared_mode": self.config.get("infrared_mode", False),
            },
        )
        self.frame_id += 1
        return packet

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
