from abc import abstractmethod

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModelFactory


class OCPBaseCroco(OCPBase):
    def __init__(
        self,
        robot_model: RobotModelFactory,
        ocp_params: OCPParamsBaseCroco,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            robot_model (RobotModelFactory): All models of the robot.
            ocp_params (OCPParamsBaseCroco): Input data structure of the OCP.
        """
        # Setting the robot model
        self._robot_model = robot_model
        self._rmodel = self._robot_model._rmodel
        self._cmodel = self._robot_model._complete_collision_model
        self._armature = self._robot_model._params.armature

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._ocp_params = ocp_params

    @property
    def horizon_size(self) -> int:
        """Number of time steps in the horizon."""
        return self._ocp_params.T

    @property
    def dt(self) -> np.float64:
        """Integration step of the OCP."""
        return self._ocp_params.dt

    @property
    @abstractmethod
    def runningModelList(self) -> list[crocoddyl.ActionModelAbstract]:
        """List of running models."""
        pass

    @property
    @abstractmethod
    def terminalModel(self) -> crocoddyl.ActionModelAbstract:
        """Terminal model."""
        pass

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
    ) -> bool:
        """Solves the OCP. Returns True if the OCP was solved successfully, False otherwise.
        The results can be accessed through the ocp_results property.

        Args:
            x0 (npt.NDArray[np.float64]): Current state of the robot.
            x_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the state. This doesn't include the current state.
            u_warmstart (list[npt.NDArray[np.float64]]): Warmstart values for the control inputs.
        """
        ### Creation of the state and actuation models

        problem = crocoddyl.ShootingProblem(
            x0, self.runningModelList, self.terminalModel
        )
        # Create solver + callbacks
        ocp = mim_solvers.SolverCSQP(problem)

        # Merit function
        ocp.use_filter_line_search = self._ocp_params.use_filter_line_search

        # Parameters of the solver
        ocp.termination_tolerance = self._ocp_params.termination_tolerance
        ocp.max_qp_iters = self._ocp_params.qp_iters
        ocp.eps_abs = self._ocp_params.eps_abs
        ocp.eps_rel = self._ocp_params.eps_rel
        ocp.with_callbacks = self._ocp_params.callbacks

        # Creating the warmstart lists for the solver
        x_init = [x0] + x_warmstart
        u_init = u_warmstart

        # Solve the OCP
        ocp.solve(x_init, u_init, self._ocp_params.solver_iters)

        # Store the results
        self.ocp_results = OCPResults(
            states=ocp.xs,
            ricatti_gains=ocp.K,
            feed_forward_terms=ocp.us,
        )

    @property
    def ocp_results(self) -> OCPResults:
        """Output data structure of the OCP.

        Returns:
            OCPResults: Output data structure of the OCP. It contains the states, Ricatti gains and feed-forward terms.
        """
        return self.ocp_results

    @property
    def debug_data(self) -> OCPDebugData:
        pass
