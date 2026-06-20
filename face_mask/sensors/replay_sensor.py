from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from face_mask.core.types import FramePacket
from face_mask.sensors.base import BaseSensor
from face_mask.utils.image_ops import to_grayscale


class ReplaySensor(BaseSensor):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.paths = [Path(path) for path in config.get("paths", [])]
        self.frame_id = 0
        self.index = 0

    def open(self) -> None:
        if not self.paths:
            raise RuntimeError("Replay sensor requires at least one frame path")

    def read(self) -> FramePacket:
        if self.index >= len(self.paths):
            if self.config.get("loop", False):
                self.index = 0
            else:
                raise StopIteration
        path = self.paths[self.index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to load replay frame: {path}")
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp_monotonic=time.monotonic(),
            image=image,
            gray=to_grayscale(image),
            metadata={"mock": False, "replay_path": str(path)},
        )
        self.frame_id += 1
        self.index += 1
        return packet

    def close(self) -> None:
        return None
