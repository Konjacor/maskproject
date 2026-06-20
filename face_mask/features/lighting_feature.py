from __future__ import annotations

import numpy as np


def compute_lighting_score(gray: np.ndarray) -> float:
    mean_value = float(np.mean(gray)) / 255.0
    centered = 1.0 - abs(mean_value - 0.45) / 0.45
    return max(0.0, min(1.0, centered))
