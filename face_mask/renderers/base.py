from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from face_mask.core.types import ExpressionOutput, FeatureBundle, PixelFrame


class BaseRenderer(ABC):
    @abstractmethod
    def render(
        self,
        expression: ExpressionOutput,
        features: FeatureBundle,
        style_config: dict[str, Any],
    ) -> PixelFrame:
        raise NotImplementedError
