from abc import abstractmethod

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt
import pinocchio as pin

from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsCrocoBase
from agimus_controller.factory.robot_model import RobotModelFactory


class OCPCrocoBase(OCPBase):
    def __init__(
        self,
        robot_model: RobotModelFactory,
        OCPParams: OCPParamsCrocoBase,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            rmodel (pin.Model): Robot model.
            cmodel (pin.GeometryModel): Collision Model of the robot.
            OCPParams (OCPParamsBase): Input data structure of the OCP.
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
        self._OCPParams = OCPParams

    @property
    def horizon_size(self) -> int:
        """Number of time steps in the horizon."""
        return self._OCPParams.T

    @property
    def dt(self) -> np.float64:
        """Integration step of the OCP."""
        return self._OCPParams.dt

    @abstractmethod
    @property
    def runningModelList(self) -> list[crocoddyl.ActionModelAbstract]:
        """List of running models."""
        ...

    @abstractmethod
    @property
    def terminalModel(self) -> crocoddyl.ActionModelAbstract:
        """Terminal model."""
        ...

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
        ocp.use_filter_line_search = self._OCPParams.use_filter_line_search

        # Parameters of the solver
        ocp.termination_tolerance = OCPParamsCrocoBase.termination_tolerance
        ocp.max_qp_iters = OCPParamsCrocoBase.qp_iters
        ocp.eps_abs = OCPParamsCrocoBase.eps_abs
        ocp.eps_rel = OCPParamsCrocoBase.eps_rel

        ocp.with_callbacks = OCPParamsCrocoBase.callbacks

        x_init = [x0] + x_warmstart
        u_init = u_warmstart

        result = ocp.solve(x_init, u_init, OCPParamsCrocoBase.solver_iters)

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
        ...
