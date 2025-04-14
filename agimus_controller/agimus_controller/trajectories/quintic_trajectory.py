import numpy as np
from typing import Tuple


class QuinticTrajectory:
    """Computes a quintic polynomial trajectory with desired amplitude and duration."""

    def __init__(self, scale_duration: np.float64, amp: np.float64):
        """Initialize polynomial attributes."""
        self.amp = amp
        self.scale_duration = scale_duration

    def get_value_at_t(
        self, t: np.float64
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """Return polynomial value and his derivatives at time t."""
        if t <= 0:
            return 0.0, 0.0, 0.0
        elif t >= self.scale_duration:
            return self.amp, 0.0, 0.0

        # Normalize time
        s = t / self.scale_duration

        polynomial = self.amp * (10 * s**3 - 15 * s**4 + 6 * s**5)
        d_polynomial = (
            self.amp * (30 * s**2 - 60 * s**3 + 30 * s**4) / self.scale_duration
        )
        dd_polynomial = (
            self.amp * (60 * s - 180 * s**2 + 120 * s**3) / (self.scale_duration**2)
        )
        return polynomial, d_polynomial, dd_polynomial
