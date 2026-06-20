from __future__ import annotations


class ExponentialSmoother:
    def __init__(self, alpha: float, initial: float | None = None) -> None:
        self.alpha = alpha
        self.value = initial

    def update(self, sample: float | None) -> float | None:
        if sample is None:
            return self.value
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value
