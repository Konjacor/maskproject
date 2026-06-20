from __future__ import annotations

import numpy as np

from face_mask.core.types import PixelFrame


ASCII_RAMP = " .:-=+*#%@"


def render_ascii(pixel_frame: PixelFrame) -> str:
    pixels = pixel_frame.pixels
    normalized = pixels.astype(np.float32) / 255.0
    lines: list[str] = []
    for row in normalized:
        chars = [ASCII_RAMP[min(len(ASCII_RAMP) - 1, int(value * (len(ASCII_RAMP) - 1)))] for value in row]
        lines.append("".join(chars))
    return "\n".join(lines)
