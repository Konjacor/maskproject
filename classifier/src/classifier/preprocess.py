from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray, grayscale: bool = True) -> np.ndarray:
    if grayscale and image.ndim == 2:
        image = image[..., None]
    image = image.astype("float32") / 255.0
    return image
