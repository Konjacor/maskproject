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
        self.frame_counter = 0

    def emit(self, expression: ExpressionOutput, features: FeatureBundle, frame: PixelFrame) -> None:
        self.frame_counter += 1
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
            print(
                f"candidate={expression.candidate_state} intensity={expression.intensity:.2f} "
                f"conf={expression.confidence:.2f} eye={_fmt(features.eye_open)} "
                f"brow={_fmt(features.brow_raise)} vis={features.visibility_score:.2f} "
                f"light={features.lighting_score:.2f} variant={expression.render_variant}"
            )
            print(ascii_face)
            self.last_state = expression.expression_state

        print_every = int(self.config.get("print_features_every", 0))
        if print_every > 0 and self.frame_counter % print_every == 0:
            print(
                f"FRAME {self.frame_counter:04d} state={expression.expression_state} "
                f"candidate={expression.candidate_state} eye={_fmt(features.eye_open)} "
                f"brow={_fmt(features.brow_raise)} conf={expression.confidence:.2f} "
                f"vis={features.visibility_score:.2f} light={features.lighting_score:.2f}"
            )


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.2f}"
