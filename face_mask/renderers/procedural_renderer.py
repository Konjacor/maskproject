from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from face_mask.core.types import ExpressionOutput, FeatureBundle, PixelFrame
from face_mask.renderers.base import BaseRenderer


class ProceduralRenderer(BaseRenderer):
    def render(
        self,
        expression: ExpressionOutput,
        features: FeatureBundle,
        style_config: dict[str, Any],
    ) -> PixelFrame:
        size = int(style_config.get("resolution", 16))
        canvas = np.zeros((size, size), dtype=np.uint8)
        eye_open = features.eye_open if features.eye_open is not None else 0.5
        brow_raise = features.brow_raise if features.brow_raise is not None else 0.0

        eye_height = max(1, int(1 + eye_open * 3))
        brow_y = max(1, min(5, int(4 - brow_raise * 2)))
        mouth_curve = int((expression.intensity - 0.2) * 6)

        cv2.rectangle(canvas, (3, 5), (5, 5 + eye_height), 255, -1)
        cv2.rectangle(canvas, (size - 6, 5), (size - 4, 5 + eye_height), 255, -1)
        cv2.line(canvas, (2, brow_y), (6, max(0, brow_y - int(brow_raise * 2))), 180, 1)
        cv2.line(canvas, (size - 7, max(0, brow_y - int(brow_raise * 2))), (size - 3, brow_y), 180, 1)

        if expression.expression_state == "surprised":
            cv2.circle(canvas, (size // 2, 11), 2 + int(expression.intensity * 2), 220, 1)
        elif expression.expression_state == "sleepy":
            cv2.line(canvas, (4, 12), (size - 4, 12), 120, 1)
        else:
            cv2.ellipse(canvas, (size // 2, 11), (4, max(1, 2 + mouth_curve)), 0, 0, 180, 220, 1)

        variant = f"{expression.expression_state}_procedural"
        expression.render_variant = variant
        return PixelFrame(pixels=canvas, metadata={"variant": variant})
