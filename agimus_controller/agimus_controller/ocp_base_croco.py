from abc import abstractmethod

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import WeightedTrajectoryPoint
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPResults, OCPDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import TrajectoryPoint


class OCPBaseCroco(OCPBase):
    def __init__(
        self,
        robot_models: RobotModels,
        params: OCPParamsBaseCroco,
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
        self.nq = self._robot_models.robot_model.nq
        self.nv = self._robot_models.robot_model.nv

        # Stat and actuation model
        self._state = crocoddyl.StateMultibody(self._robot_models.robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Setting the OCP parameters
        self._params = params
        self._solver = None
        self._ocp_results: OCPResults = None
        self._debug_data: OCPDebugData = OCPDebugData(
            problem_solved=None,
            result=None,
            references=None,
            kkt_norm=None,
            collision_distance_residuals=None,
        )

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
        self._problem.nthreads = self._params.nb_threads

        # Create solver + callbacks
        self._solver = mim_solvers.SolverCSQP(self._problem)

        # Merit function
        self._solver.use_filter_line_search = self._params.use_filter_line_search

        # Parameters of the solver
        self._solver.termination_tolerance = self._params.termination_tolerance
        self._solver.max_qp_iters = self._params.qp_iters
        self._solver.eps_abs = self._params.eps_abs
        self._solver.eps_rel = self._params.eps_rel
        if self._params.callbacks:
            self._solver.setCallbacks(
                [mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()]
            )

    @property
    def horizon_size(self) -> int:
        """Number of time steps in the horizon."""
        return self._params.horizon_size

    @property
    def dt(self) -> float:
        """Integration step of the OCP."""
        return self._params.dt

    @property
    def problem(self) -> crocoddyl.ShootingProblem:
        return self._problem

    def set_reference_weighted_trajectory(
        self, reference_weighted_trajectory: list[WeightedTrajectoryPoint]
    ):
        """Set the reference trajectory for the OCP."""
        reference_trajectory_points = [el.point for el in reference_weighted_trajectory]
        self._debug_data.references = reference_trajectory_points

    @abstractmethod
    def create_running_model_list(self) -> list[crocoddyl.ActionModelAbstract]:
        """Create the list of running models."""
        pass

    @abstractmethod
    def create_terminal_model(self) -> crocoddyl.ActionModelAbstract:
        """Create the terminal model."""
        pass

    def modify_cost_reference_and_weights(
        self,
        model: crocoddyl.ActionModelAbstract,
        cost_name: str,
        reference: npt.NDArray[np.float64],
        weigths: npt.NDArray[np.float64],
    ):
        """modify crocoddyl cost reference and weight."""
        model.differential.costs.costs[cost_name].cost.residual.reference = reference
        model.differential.costs.costs[cost_name].cost.activation.weights = weigths

    def solve(
        self,
        x0: npt.NDArray[np.float64],
        x_warmstart: list[npt.NDArray[np.float64]],
        u_warmstart: list[npt.NDArray[np.float64]],
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
        res = self._solver.solve(
            [x0] + x_warmstart, u_warmstart, self._params.solver_iters
        )
        solution = [
            TrajectoryPoint(
                time_ns=-1,
                robot_configuration=state[: self.nq],
                robot_velocity=state[self.nq :],
                robot_acceleration=np.zeros_like(state[self.nq :]),
            )
            for state in self._solver.xs
        ]
        self._debug_data.problem_solved = res
        self._debug_data.kkt_norm = self._solver.KKT
        self._debug_data.result = solution

        # Store the results
        self._ocp_results = OCPResults(
            states=self._solver.xs,
            ricatti_gains=self._solver.K,
            feed_forward_terms=self._solver.us,
        )

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
