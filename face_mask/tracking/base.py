from __future__ import annotations

from abc import ABC, abstractmethod

from face_mask.core.types import FramePacket, TrackingResult


class BaseTracker(ABC):
    @abstractmethod
    def track(self, frame: FramePacket) -> TrackingResult:
        raise NotImplementedError
