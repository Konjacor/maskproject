from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from face_mask.core.types import ExpressionOutput, FeatureBundle, PixelFrame
from face_mask.renderers.base import BaseRenderer


class PresetRenderer(BaseRenderer):
    def render(
        self,
        expression: ExpressionOutput,
        features: FeatureBundle,
        style_config: dict[str, Any],
    ) -> PixelFrame:
        size = int(style_config.get("resolution", 16))
        canvas = np.zeros((size, size), dtype=np.uint8)
        state = expression.expression_state

        if state == "neutral":
            cv2.line(canvas, (4, 11), (size - 4, 11), 180, 1)
        elif state == "cheerful":
            cv2.ellipse(canvas, (size // 2, 10), (5, 3), 0, 0, 180, 220, 1)
        elif state == "surprised":
            cv2.circle(canvas, (size // 2, 11), 3, 220, 1)
        elif state == "sleepy":
            cv2.line(canvas, (4, 8), (7, 9), 180, 1)
            cv2.line(canvas, (size - 8, 9), (size - 5, 8), 180, 1)

        eye_level = 4 if state == "sleepy" else 6
        cv2.line(canvas, (4, eye_level), (6, eye_level), 255, 1)
        cv2.line(canvas, (size - 7, eye_level), (size - 5, eye_level), 255, 1)

        variant = f"{state}_preset"
        expression.render_variant = variant
        return PixelFrame(pixels=canvas, metadata={"variant": variant})
