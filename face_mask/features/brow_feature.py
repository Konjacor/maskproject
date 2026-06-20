from __future__ import annotations

import numpy as np


def compute_brow_raise_score(brow_roi: np.ndarray, eye_roi: np.ndarray | None = None) -> float:
    if brow_roi.size == 0:
        return 0.0
    brow_mean = float(np.mean(brow_roi)) / 255.0
    brow_gradient = float(np.mean(np.abs(np.diff(brow_roi.astype(np.float32), axis=0)))) / 32.0
    if eye_roi is None or eye_roi.size == 0:
        baseline = 0.45
    else:
        baseline = float(np.mean(eye_roi)) / 255.0
    relative = (baseline - brow_mean) * 1.8 + brow_gradient
    return max(-1.0, min(1.0, relative))
