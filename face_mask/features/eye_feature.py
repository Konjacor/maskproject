from __future__ import annotations

import numpy as np


def compute_eye_open_score(eye_roi: np.ndarray) -> float:
    if eye_roi.size == 0:
        return 0.0
    normalized = eye_roi.astype(np.float32) / 255.0
    vertical_profile = normalized.mean(axis=1)
    spread = float(np.max(vertical_profile) - np.min(vertical_profile))
    edge_change = float(np.mean(np.abs(np.diff(vertical_profile))))
    score = 0.65 * spread + 0.35 * edge_change * 4.0
    return max(0.0, min(1.0, score))
