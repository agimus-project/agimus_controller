import numpy as np
import numpy.typing as npt
from typing import Tuple


class QuinticTrajectory:
    """Computes a quintic polynomial trajectory with desired amplitude and duration."""

    def __init__(self, scale_duration: npt.NDArray[np.float64]):
        """Initialize polynomial attributes."""

        self.scale_duration = scale_duration
        self.p = np.zeros(self.scale_duration.size)
        self.v = np.zeros(self.scale_duration.size)
        self.a = np.zeros(self.scale_duration.size)

    def get_value_at_t(
        self, t: np.float64
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Return polynomial value and his derivatives at time t."""
        for i in range(self.scale_duration.size):
            if t <= 0:
                self.p[i] = 0.0
                self.v[i] = 0.0
                self.a[i] = 0.0
                continue
            elif t >= self.scale_duration[i]:
                self.p[i] = 1.0
                self.v[i] = 0.0
                self.a[i] = 0.0
                continue

            # Normalize time
            s = t / self.scale_duration[i]
            self.p[i] = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.v[i] = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.scale_duration[i]
            self.a[i] = (60 * s - 180 * s**2 + 120 * s**3) / (
                self.scale_duration[i] ** 2
            )

        return self.p, self.v, self.a
