from __future__ import annotations

import logging
from pathlib import Path

import cv2

from face_mask.core.logging_setup import log_event
from face_mask.core.types import ExpressionOutput, FeatureBundle, FramePacket, PixelFrame, TrackingResult
from face_mask.renderers.ascii_renderer import render_ascii


class DebugOutput:
    def __init__(self, logger: logging.Logger, config: dict) -> None:
        self.logger = logger
        self.config = config
        self.last_state: str | None = None
        self.frame_counter = 0
        self.snapshot_dir = Path(config.get("snapshot_dir", "debug_frames"))

    def emit(
        self,
        expression: ExpressionOutput,
        features: FeatureBundle,
        frame: PixelFrame,
        source_frame: FramePacket,
        tracking: TrackingResult,
    ) -> None:
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
                roi_stats=features.raw.get("roi_stats", {}),
                tracking_notes=features.raw.get("tracking_notes", []),
            )
            print(f"\nSTATE: {expression.expression_state}")
            print(
                f"candidate={expression.candidate_state} intensity={expression.intensity:.2f} "
                f"conf={expression.confidence:.2f} eye={_fmt(features.eye_open)} "
                f"brow={_fmt(features.brow_raise)} vis={features.visibility_score:.2f} "
                f"light={features.lighting_score:.2f} variant={expression.render_variant}"
            )
            _print_roi_stats(features)
            print(ascii_face)
            self.last_state = expression.expression_state

        print_every = int(self.config.get("print_features_every", 0))
        snapshot_every = int(self.config.get("snapshot_every", 0))
        if print_every > 0 and self.frame_counter % print_every == 0:
            print(
                f"FRAME {self.frame_counter:04d} state={expression.expression_state} "
                f"candidate={expression.candidate_state} eye={_fmt(features.eye_open)} "
                f"brow={_fmt(features.brow_raise)} conf={expression.confidence:.2f} "
                f"vis={features.visibility_score:.2f} light={features.lighting_score:.2f}"
            )
            _print_roi_stats(features)

        if snapshot_every > 0 and self.frame_counter % snapshot_every == 0:
            self._save_snapshot(source_frame, tracking, expression, features)

    def _save_snapshot(
        self,
        source_frame: FramePacket,
        tracking: TrackingResult,
        expression: ExpressionOutput,
        features: FeatureBundle,
    ) -> None:
        image = source_frame.image.copy()
        for name, roi in tracking.visible_regions.items():
            cv2.rectangle(image, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (0, 255, 0), 2)
            cv2.putText(
                image,
                name,
                (roi.x, max(15, roi.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        header = (
            f"state={expression.expression_state} cand={expression.candidate_state} "
            f"eye={_fmt(features.eye_open)} brow={_fmt(features.brow_raise)} "
            f"conf={expression.confidence:.2f} vis={features.visibility_score:.2f} "
            f"light={features.lighting_score:.2f}"
        )
        cv2.putText(image, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"frame_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(path), image)


def _print_roi_stats(features: FeatureBundle) -> None:
    roi_stats = features.raw.get("roi_stats", {})
    tracking_notes = features.raw.get("tracking_notes", [])
    if tracking_notes:
        print(f"notes={','.join(tracking_notes)}")
    for name in ("left_eye", "right_eye", "brow"):
        stats = roi_stats.get(name)
        if not stats:
            continue
        print(
            f"  {name}: x={stats['x']} y={stats['y']} w={stats['width']} h={stats['height']} "
            f"mean={stats['mean']:.1f} std={stats['std']:.1f} min={stats['min']} max={stats['max']}"
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.2f}"
