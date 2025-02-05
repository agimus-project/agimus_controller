"""Implement OCPBase."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPBase(ABC):
    """Base class for the Optimal Control Problem (OCP) solver.

    This class defines the interface for the OCP solver.
    """

    def __init__(self) -> None:
        """Construct default object."""
        pass

    @abstractmethod
    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ) -> None:
        """Set the reference trajectory and the weights of the costs for the OCP solver.

        This method should be implemented by the derived class.
        """
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
    def dt() -> float:
        """Returns the time step of the OCP in seconds.

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
        """Solve the OCP.

        This method should be implemented by the derived class.
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
        """Return the results of the OCP solver.

        The solve method should be called before calling this method.

        Returns:
            OCPResults: Class containing the results of the OCP solver.

        """
        pass

    @ocp_results.setter
    def ocp_results(self, value: OCPResults) -> None:
        """Set the output data structure of the OCP.

        Args:
            value (OCPResults): New output data structure of the OCP.

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
