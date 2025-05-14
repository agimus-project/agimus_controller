import numpy as np


class WeightIncreasing:
    """
    Class that defines a weight increasing over time using an hyperbolic tangent shape.
    """

    def __init__(self, max_weight: float, percent: float, time_reach_percent: float):
        """
        Initialize the increasing weigth parameters.
        """
        self.max_weight = max_weight
        self.percent = percent
        self.time_reach_percent = time_reach_percent

    def get_weight_at_t(self, t):
        return self.max_weight * np.tanh(
            t * np.arctanh(self.percent) / self.time_reach_percent
        )
