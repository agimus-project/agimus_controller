from abc import abstractmethod

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt

from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import WeightedTrajectoryPoint


class OCPBaseCroco(OCPBase):
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
    ) -> None:
        """Defines common behavior for all OCP using croccodyl. This is an abstract class with some helpers to design OCPs in a more friendly way.

        Args:
            robot_models (RobotModels): All models of the robot.
            ocp_params (OCPParamsBaseCroco): Input data structure of the OCP.
        """
        # Setting the robot model
        self._robot_models = robot_models
        self._collision_model = self._robot_models.collision_model
        self._armature = self._robot_models.armature

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._robot_models.robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._ocp_params = ocp_params
        self._solver = None
        self._ocp_results: OCPResults = None
        self._debug_data: OCPDebugData = OCPDebugData()

        # Create the running models
        self._running_model_list = self.create_running_model_list()
        # Create the terminal model
        self._terminal_model = self.create_terminal_model()
        # Create the shooting problem
        self._problem = crocoddyl.ShootingProblem(
            np.zeros(
                self._robot_models.robot_model.nq + self._robot_models.robot_model.nv
            ),
            self._running_model_list,
            self._terminal_model,
        )
        # Create solver + callbacks
        self._solver = mim_solvers.SolverCSQP(self._problem)

        # Merit function
        self._solver.use_filter_line_search = self._ocp_params.use_filter_line_search

        # Parameters of the solver
        if self._ocp_params.max_solve_time is not None:
            self._solver.max_solve_time = self._ocp_params.max_solve_time
        self._solver.termination_tolerance = self._ocp_params.termination_tolerance
        self._solver.max_qp_iters = self._ocp_params.qp_iters
        self._solver.eps_abs = self._ocp_params.eps_abs
        self._solver.eps_rel = self._ocp_params.eps_rel
        if self._ocp_params.callbacks:
            self._solver.setCallbacks(
                [mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()]
            )

    @property
    def n_controls(self) -> int:
        """Number of controls in the OCP."""
        return self._ocp_params.n_controls

    @property
    def dt(self) -> float:
        """Initial integration step of the OCP."""
        return self._ocp_params.dt

    @property
    def problem(self) -> crocoddyl.ShootingProblem:
        return self._problem

    @abstractmethod
    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        """Create the list of running models."""
        pass

    @abstractmethod
    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        """Create the terminal model."""
        pass

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""
        if self._ocp_params.use_debug_data:
            reference_trajectory_points = [
                el.point for el in reference_weighted_trajectory
            ]
            self._debug_data.references = reference_trajectory_points

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
        use_iteration_limits_and_timeout: bool = True,
    ) -> None:
        """Solves the OCP.
        The results can be accessed through the ocp_results property.

        Args:
            x0 (npt.NDArray[np.float64]): Current state of the robot.
            x_warmstart (list[npt.NDArray[np.float64]]): Predicted states for the OCP.
            u_warmstart (list[npt.NDArray[np.float64]]): Predicted control inputs for the OCP.
        """
        # Set the initial state
        self._problem.x0 = x0
        # Solve the OCP
        max_iters = (
            self._ocp_params.solver_iters if use_iteration_limits_and_timeout else 1000
        )
        # This if is only meant as a way of smoothing the integration of attribute
        # `SolverCSQP.max_solve_time`. If the user did not set `max_solve_time`, then
        # don't touch the attribute.
        if self._ocp_params.max_solve_time is not None:
            self._solver.max_solve_time = (
                self._ocp_params.max_solve_time
                if use_iteration_limits_and_timeout
                else float("inf")
            )
        res = self._solver.solve(x_warmstart, u_warmstart, max_iters)
        ocp_results = OCPResults(
            states=self._solver.xs,
            ricatti_gains=self._solver.K,
            feed_forward_terms=self._solver.us,
        )
        if self._ocp_params.use_debug_data:
            self._debug_data.problem_solved = res
            self._debug_data.result = ocp_results
            self._debug_data.kkt_norm = self._solver.KKT
            self._debug_data.nb_iter = int(self._solver.iter)
            self._debug_data.nb_qp_iter = int(self._solver.qp_iters)
            if len(self._debug_data.residuals) == 0:
                names = (
                    self._problem.runningModels[0]
                    .differential.costs.costs.todict()
                    .keys()
                )
            else:
                names = self._debug_data.residuals.keys()
            self._debug_data.residuals = {name: [] for name in names}

            for data in self._problem.runningDatas:
                for name in names:
                    self._debug_data.residuals[name].append(
                        data.differential.costs.costs[name].residual.r
                    )

        # Store the results
        self._ocp_results = ocp_results

    def integrate(
        self, state: npt.NDArray[np.float64], control: npt.NDArray
    ) -> npt.NDArray[np.float64]:
        data = self._problem.runningDatas[0]
        self._problem.runningModels[0].calc(data, state, control)
        return data.xnext

    @property
    def ocp_results(self) -> OCPResults:
        """Output data structure of the OCP.

        Returns:
            OCPResults: Output data structure of the OCP. It contains the states, Ricatti gains, and feed-forward terms.
        """
        return self._ocp_results

    @ocp_results.setter
    def ocp_results(self, value: OCPResults) -> None:
        """Set the output data structure of the OCP.

        Args:
            value (OCPResults): New output data structure of the OCP.
        """
        self._ocp_results = value

    @property
    def debug_data(self) -> OCPDebugData:
        return self._debug_data

    @debug_data.setter
    def debug_data(self, value: OCPDebugData) -> None:
        self._debug_data = value
