from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def clamp(self, image_width: int, image_height: int) -> "ROI":
        x = max(0, min(self.x, image_width - 1))
        y = max(0, min(self.y, image_height - 1))
        width = max(1, min(self.width, image_width - x))
        height = max(1, min(self.height, image_height - y))
        return ROI(x=x, y=y, width=width, height=height)

    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def translate(self, dx: int, dy: int) -> "ROI":
        return ROI(x=self.x + dx, y=self.y + dy, width=self.width, height=self.height)

    def scale_from_center(self, scale_x: float, scale_y: float) -> "ROI":
        center_x, center_y = self.center()
        width = max(1, int(self.width * scale_x))
        height = max(1, int(self.height * scale_y))
        return ROI(
            x=center_x - width // 2,
            y=center_y - height // 2,
            width=width,
            height=height,
        )


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    timestamp_monotonic: float
    image: np.ndarray
    gray: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackingResult:
    tracking_mode: str
    visible_regions: dict[str, ROI]
    visibility_score: float
    face_present: bool
    lighting_score: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureBundle:
    eye_open: float | None
    brow_raise: float | None
    mouth_open: float | None = None
    asymmetry: float | None = None
    motion_level: float | None = None
    visibility_score: float = 1.0
    lighting_score: float = 1.0
    confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpressionOutput:
    expression_state: str
    intensity: float
    confidence: float
    render_variant: str | None = None
    debug_tags: list[str] = field(default_factory=list)
    candidate_state: str | None = None


@dataclass(slots=True)
class PixelFrame:
    pixels: np.ndarray
    brightness: int = 255
    metadata: dict[str, Any] = field(default_factory=dict)
