from __future__ import annotations

import time
from typing import Any

from face_mask.core.smoothing import ExponentialSmoother
from face_mask.core.types import ExpressionOutput, FeatureBundle


class ExpressionStateMachine:
    def __init__(self, config: dict[str, Any]) -> None:
        smoothing_config = config.get("features", {})
        state_config = config.get("state_machine", {})
        self.eye_smoother = ExponentialSmoother(smoothing_config.get("eye_smoothing_alpha", 0.45))
        self.brow_smoother = ExponentialSmoother(smoothing_config.get("brow_smoothing_alpha", 0.25))
        self.min_hold_seconds = state_config.get("min_hold_ms", 180) / 1000.0
        self.sleepy_enter = state_config.get("sleepy_enter", 0.28)
        self.sleepy_exit = state_config.get("sleepy_exit", 0.38)
        self.surprised_brow_enter = state_config.get("surprised_brow_enter", 0.45)
        self.surprised_brow_exit = state_config.get("surprised_brow_exit", 0.28)
        self.cheerful_brow_enter = state_config.get("cheerful_brow_enter", 0.18)
        self.current_state = "neutral"
        self.last_switch_at = 0.0

    def update(self, bundle: FeatureBundle) -> tuple[FeatureBundle, ExpressionOutput]:
        smoothed_eye = self.eye_smoother.update(bundle.eye_open)
        smoothed_brow = self.brow_smoother.update(bundle.brow_raise)
        bundle.eye_open = smoothed_eye
        bundle.brow_raise = smoothed_brow

        candidate_state = self._candidate_state(bundle)
        now = time.monotonic()
        if candidate_state != self.current_state and (now - self.last_switch_at) >= self.min_hold_seconds:
            self.current_state = candidate_state
            self.last_switch_at = now

        intensity = self._compute_intensity(bundle)
        debug_tags: list[str] = []
        if bundle.visibility_score < 0.35:
            debug_tags.append("low_visibility")
        if bundle.lighting_score < 0.3:
            debug_tags.append("lighting_instability")
        if bundle.confidence < 0.4:
            debug_tags.append("low_confidence")

        return bundle, ExpressionOutput(
            expression_state=self.current_state,
            intensity=intensity,
            confidence=bundle.confidence,
            candidate_state=candidate_state,
            debug_tags=debug_tags,
        )

    def _candidate_state(self, bundle: FeatureBundle) -> str:
        eye = bundle.eye_open if bundle.eye_open is not None else 0.5
        brow = bundle.brow_raise if bundle.brow_raise is not None else 0.0

        if eye < self.sleepy_enter or (self.current_state == "sleepy" and eye < self.sleepy_exit):
            return "sleepy"
        if eye > 0.70 and (brow > self.surprised_brow_enter or (self.current_state == "surprised" and brow > self.surprised_brow_exit)):
            return "surprised"
        if eye > 0.45 and brow > self.cheerful_brow_enter:
            return "cheerful"
        return "neutral"

    def _compute_intensity(self, bundle: FeatureBundle) -> float:
        eye = bundle.eye_open if bundle.eye_open is not None else 0.5
        brow = bundle.brow_raise if bundle.brow_raise is not None else 0.0
        confidence = bundle.confidence
        raw = max(abs(eye - 0.5) * 1.2, abs(brow))
        return max(0.0, min(1.0, raw * confidence))
