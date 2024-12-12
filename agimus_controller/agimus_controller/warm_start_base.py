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
        initial_state: TrajectoryPoint,
        reference_trajectory: list[TrajectoryPoint],
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
    ]:
        """
        Generate initial values for a warm-start of the optimization problem.

        Args:
            initial_state (TrajectoryPoint): The initial state of the robot,
                containing `robot_configuration` and `robot_velocity`.
            reference_trajectory (list[TrajectoryPoint]): A list of `TrajectoryPoint` objects
                representing the reference trajectory.

        Returns:
            tuple:
                - x0 (npt.NDArray[np.float64]): The initial state vector.
                - init_xs (list[npt.NDArray[np.float64]]): List of state vectors
                for each point in the reference trajectory.
                - init_us (list[npt.NDArray[np.float64]]): List of control inputs.
        """
        ...

    def update_previous_solution(
        self, previous_solution: list[TrajectoryPoint]
    ) -> None:
        """Stores internally the previous solution of the OCP"""
        self._previous_solution = previous_solution
