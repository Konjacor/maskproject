from __future__ import annotations

import logging

from face_mask.core.logging_setup import log_event
from face_mask.core.types import ExpressionOutput, FeatureBundle, PixelFrame
from face_mask.renderers.ascii_renderer import render_ascii


class DebugOutput:
    def __init__(self, logger: logging.Logger, config: dict) -> None:
        self.logger = logger
        self.config = config
        self.last_state: str | None = None

    def emit(self, expression: ExpressionOutput, features: FeatureBundle, frame: PixelFrame) -> None:
        if expression.expression_state != self.last_state or self.config.get("always_print_ascii", False):
            ascii_face = render_ascii(frame)
            log_event(
                self.logger,
                logging.INFO,
                "state.transition",
                state=expression.expression_state,
                candidate_state=expression.candidate_state,
                confidence=expression.confidence,
                eye_open=features.eye_open,
                brow_raise=features.brow_raise,
                render_variant=expression.render_variant,
            )
            print(f"\nSTATE: {expression.expression_state}")
            print(ascii_face)
            self.last_state = expression.expression_state
