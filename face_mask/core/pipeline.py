from __future__ import annotations

import logging
import time
from typing import Any

from face_mask.core.logging_setup import log_event
from face_mask.core.state_machine import ExpressionStateMachine
from face_mask.features.fusion import build_feature_bundle
from face_mask.output.debug_output import DebugOutput
from face_mask.renderers.preset_renderer import PresetRenderer
from face_mask.renderers.procedural_renderer import ProceduralRenderer
from face_mask.sensors.camera_sensor import CameraSensor
from face_mask.sensors.mock_sensor import MockSensor
from face_mask.sensors.replay_sensor import ReplaySensor
from face_mask.tracking.full_face_tracker import FullFaceTracker
from face_mask.tracking.upper_face_tracker import UpperFaceTracker
from face_mask.utils.timing import timed


class RuntimePipeline:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.sensor = self._build_sensor(config)
        self.tracker = self._build_tracker(config)
        self.state_machine = ExpressionStateMachine(config)
        self.renderer = self._build_renderer(config)
        self.output = DebugOutput(logger, config.get("output", {}))
        self.frame_delay = 1.0 / max(1, int(config.get("runtime", {}).get("fps_target", 20)))

    def run(self) -> None:
        runtime_config = self.config.get("runtime", {})
        max_frames = runtime_config.get("max_frames")
        self.sensor.open()
        log_event(self.logger, logging.INFO, "runtime.startup", config=self.config)

        try:
            frames_processed = 0
            while max_frames is None or frames_processed < max_frames:
                with timed() as elapsed_ms:
                    frame = self.sensor.read()
                    tracking = self.tracker.track(frame)
                    features = build_feature_bundle(frame, tracking)
                    features, expression = self.state_machine.update(features)
                    pixel_frame = self.renderer.render(expression, features, self.config.get("renderer", {}))
                    self.output.emit(expression, features, pixel_frame, frame, tracking)
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "features.values",
                        frame_id=frame.frame_id,
                        tracking_mode=tracking.tracking_mode,
                        visibility_score=tracking.visibility_score,
                        lighting_score=tracking.lighting_score,
                        eye_open=features.eye_open,
                        brow_raise=features.brow_raise,
                        confidence=features.confidence,
                        render_variant=expression.render_variant,
                    )
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "perf.frame_time",
                        frame_id=frame.frame_id,
                        elapsed_ms=elapsed_ms(),
                    )
                frames_processed += 1
                time.sleep(self.frame_delay)
        except StopIteration:
            log_event(self.logger, logging.INFO, "runtime.replay_complete")
        finally:
            self.sensor.close()
            log_event(self.logger, logging.INFO, "runtime.shutdown")

    def _build_sensor(self, config: dict[str, Any]):
        sensor_config = config.get("sensor", {})
        sensor_type = sensor_config.get("type", "mock")
        if sensor_type == "camera":
            return CameraSensor(sensor_config)
        if sensor_type == "replay":
            return ReplaySensor(sensor_config)
        return MockSensor(sensor_config)

    def _build_tracker(self, config: dict[str, Any]):
        tracking_mode = config.get("runtime", {}).get("tracking_mode", "upper_face")
        if tracking_mode == "full_face":
            return FullFaceTracker(tracking_mode=tracking_mode)
        return UpperFaceTracker(tracking_mode=tracking_mode)

    def _build_renderer(self, config: dict[str, Any]):
        render_mode = config.get("runtime", {}).get("render_mode", "procedural")
        if render_mode == "preset":
            return PresetRenderer()
        return ProceduralRenderer()
