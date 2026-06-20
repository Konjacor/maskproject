from __future__ import annotations

import numpy as np


def compute_visibility_score(gray: np.ndarray) -> float:
    contrast = float(np.std(gray)) / 64.0
    return max(0.0, min(1.0, contrast))
