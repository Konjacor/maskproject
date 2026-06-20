from __future__ import annotations

from abc import ABC, abstractmethod

from face_mask.core.types import FramePacket


class BaseSensor(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> FramePacket:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
