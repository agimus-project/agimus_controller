from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import TrajectoryPoint


class WarmStartBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._previous_solution: list[TrajectoryPoint] = list()

    @abstractmethod
    def generate(
        self,
        reference_trajectory: list[TrajectoryPoint],
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
    ]:
        """Returns x0, x_init, u_init."""
        ...

    def update_previous_solution(
        self, previous_solution: list[TrajectoryPoint]
    ) -> None:
        """Stores internally the previous solution of the OCP"""
        self._previous_solution = previous_solution
