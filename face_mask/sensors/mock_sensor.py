from __future__ import annotations

import math
import random
import time
from typing import Any

import cv2
import numpy as np

from face_mask.core.types import FramePacket
from face_mask.sensors.base import BaseSensor
from face_mask.utils.image_ops import to_grayscale


class MockSensor(BaseSensor):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.frame_id = 0
        self.started_at = 0.0
        self.random = random.Random(config.get("seed", 7))

    def open(self) -> None:
        self.started_at = time.monotonic()

    def read(self) -> FramePacket:
        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 480))
        elapsed = time.monotonic() - self.started_at
        profile = self.config.get("profile", "wave_features")
        image = np.zeros((height, width, 3), dtype=np.uint8)

        base_brightness = int(self.config.get("base_brightness", 48))
        image[:] = base_brightness

        eye_open = 0.5 + 0.35 * math.sin(elapsed * self.config.get("eye_speed", 1.8))
        brow_raise = 0.4 * math.sin(elapsed * self.config.get("brow_speed", 0.9) + 0.8)

        if profile == "cycle_states":
            phase = int(elapsed // self.config.get("state_period_seconds", 2.0)) % 4
            cycle = [(0.45, 0.0), (0.72, 0.4), (0.92, 0.8), (0.18, -0.2)]
            eye_open, brow_raise = cycle[phase]

        noise = self.config.get("noise", 0.03)
        eye_open = max(0.0, min(1.0, eye_open + self.random.uniform(-noise, noise)))
        brow_raise = max(-1.0, min(1.0, brow_raise + self.random.uniform(-noise, noise)))

        self._draw_mock_face(image, eye_open, brow_raise)

        if self.config.get("infrared_mode", False):
            image[:, :, 1] = 0
            image[:, :, 2] = 0

        gray = to_grayscale(image)
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp_monotonic=time.monotonic(),
            image=image,
            gray=gray,
            metadata={
                "mock": True,
                "profile": profile,
                "eye_open": eye_open,
                "brow_raise": brow_raise,
            },
        )
        self.frame_id += 1
        return packet

    def close(self) -> None:
        return None

    def _draw_mock_face(self, image: np.ndarray, eye_open: float, brow_raise: float) -> None:
        height, width = image.shape[:2]
        eye_y = int(height * 0.34)
        left_eye_x = int(width * 0.30)
        right_eye_x = int(width * 0.70)
        brow_y = int(height * (0.22 - brow_raise * 0.04))
        eye_half_height = max(2, int(3 + eye_open * 14))
        eye_half_width = max(12, int(width * 0.07))

        cv2.ellipse(image, (left_eye_x, eye_y), (eye_half_width, eye_half_height), 0, 0, 360, (235, 235, 235), -1)
        cv2.ellipse(image, (right_eye_x, eye_y), (eye_half_width, eye_half_height), 0, 0, 360, (235, 235, 235), -1)
        cv2.line(image, (left_eye_x - eye_half_width, brow_y), (left_eye_x + eye_half_width, brow_y - int(brow_raise * 8)), (180, 180, 180), 4)
        cv2.line(image, (right_eye_x - eye_half_width, brow_y - int(brow_raise * 8)), (right_eye_x + eye_half_width, brow_y), (180, 180, 180), 4)

        if self.config.get("hotspots", False):
            cv2.circle(image, (int(width * 0.18), int(height * 0.18)), 32, (90, 90, 90), -1)
            cv2.circle(image, (int(width * 0.82), int(height * 0.20)), 28, (110, 110, 110), -1)
