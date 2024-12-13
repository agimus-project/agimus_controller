from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import TrajectoryPoint


class WarmStartBase(ABC):
    """
    A template class for implementing methods that generate a warmstart for an optimal control problem.
    The warmstart should generate the initial values for state and control trajectories,
    based on the initial robot state and a reference trajectory.

    Attributes:
        _previous_solution (list[TrajectoryPoint]): Stores the previous solution of the optimization problem.

    Methods:
        generate(initial_state: TrajectoryPoint, reference_trajectory: list[TrajectoryPoint]) -> tuple:
            Generates warm-start values for the optimization problem. This must be
            implemented by subclasses.

        update_previous_solution(previous_solution: list[TrajectoryPoint]) -> None:
            Updates the internally stored previous solution for later use.
    """

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
        """
        This method stores the solution from a previous optimization cycle.
        It can be used as a reference or initialization of warmstart.

        Args:
            previous_solution (list[TrajectoryPoint]): The solution of the optimization problem.
        """
        self._previous_solution = previous_solution
