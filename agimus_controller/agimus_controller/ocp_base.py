from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPBase(ABC):
    """Base class for the Optimal Control Problem (OCP) solver. This class defines the interface for the OCP solver.
    If you want to implement a new OCP solver, you should derive from this class and implement the abstract methods.
    If you want to use Crocoddyl, you should inherit from the OCPCrocoBase class instead.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_reference_horizon(
        self, reference_trajectory: list[WeightedTrajectoryPoint]
    ) -> None:
        """Set the reference trajectory and the weights of the costs for the OCP solver. This method should be implemented by the derived class."""
        pass

    @property
    @abstractmethod
    def horizon_size() -> int:
        """Returns the horizon size of the OCP.

        Returns:
            int: size of the horizon.
        """
        pass

    @property
    @abstractmethod
    def dt() -> int:
        """Returns the time step of the OCP.

        Returns:
            int: time step of the OCP.
        """
        pass

    @abstractmethod
    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
    ) -> None:
        """Solver for the OCP. This method should be implemented by the derived class.
        The method should solve the OCP for the given initial state and warmstart values.

        Args:
            x0 (npt.NDArray[np.float64]): current state of the robot.
            x_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the state. This doesn't include the current state.
            u_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the control inputs.
        """
        pass

    @property
    @abstractmethod
    def ocp_results(self) -> OCPResults:
        """Returns the results of the OCP solver.
        The solve method should be called before calling this method.

        Returns:
            OCPResults: Class containing the results of the OCP solver.
        """
        pass

    @property
    @abstractmethod
    def debug_data(self) -> OCPDebugData:
        """Returns the debug data of the OCP solver.

        Returns:
            OCPDebugData: Class containing the debug data of the OCP solver.
        """
        pass
