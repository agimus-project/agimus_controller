"""WarmStartBase Module.

This module defines the WarmStartBase class, a template for generating warm-starts
for optimal control problems. It includes methods to initialize state and control
trajectories based on an initial robot state and a reference trajectory.

Example:
    Subclass the WarmStartBase class and implement the `generate` method to create
    a warm-start for a specific optimization problem.

"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import typing as T

from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.mpc_data import OCPResults


class WarmStartBase(ABC):
    """Base class for generating warm-starts for optimal control problems.

    This class provides a template for generating initial values for state and
    control trajectories based on the initial robot state and a reference trajectory.
    """

    def __init__(self) -> None:
        """Initialize the WarmStartBase class."""
        super().__init__()
        # Stores the previous solution of the optimization problem.
        self._previous_solution: T.Optional[OCPResults] = None

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
        """Generate initial values for a warm-start of the optimization problem.

        Args:
            initial_state (TrajectoryPoint): The initial state of the robot,
                containing `robot_configuration` and `robot_velocity`.
            reference_trajectory (list[TrajectoryPoint]): A list of `TrajectoryPoint` objects
                representing the reference trajectory.

        Returns:
            tuple:
                - npt.NDArray[np.float64]: The initial state vector.
                - list[npt.NDArray[np.float64]]: List of state vectors
                for each point in the reference trajectory.
                - list[npt.NDArray[np.float64]]: List of control inputs.

        """
        ...

    @abstractmethod
    def setup(self, *args, **kwargs) -> None:
        """Set up the variables needed for the warmstart computation. Allows to pass additional variables after the class is initialised, that are only know at the runtime.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Example:
            >>> class MyPinocchioWarmstart(WarmStartBase):
            ...     def setup(self, rmodel: pin.Model) -> None:
            ...         self._rmodel = rmodel
            ...         self._rdata = self._rmodel.createData()
            ...         self._nx = self._rmodel.nq + self._rmodel.nv
            >>> warmstart = MyPinocchioWarmstart()
            >>> # ...
            >>> rmodel: pin.Model() = await_urdf_model()
            >>> warmstart.setup(rmodel)

        """
        ...

    def update_previous_solution(self, previous_solution: OCPResults) -> None:
        """Update the stored previous solution.

        Stores the solution from a previous optimization cycle to be used as a reference
        or initialization for warm-start generation.

        Args:
            previous_solution (OCPResuls): The solution of the optimization problem.
        """
        self._previous_solution = previous_solution
