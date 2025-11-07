import typing as T
import yaml

import crocoddyl
import mim_solvers
import numpy as np
import numpy.typing as npt
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.mpc_data import OCPDebugData, OCPResults
from agimus_controller.ocp.ocp_croco_generic import (
    OCPCrocoGeneric,
    ShootingProblem,
)
from agimus_controller.ocp_param_base import OCPParamsBaseCroco


class OCPCrocoForceFeedbackGeneric(OCPCrocoGeneric):
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
        yaml_file: T.Union[str, T.IO],
        expect_rolling_buffer: bool = False,
    ) -> None:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        self._data = ShootingProblem(**data)
        self._enabled_directions = (
            self._data.running_model.differential.enabled_directions
        )
        self._expect_rolling_buffer = expect_rolling_buffer

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
                self._robot_models.robot_model.nq
                + self._robot_models.robot_model.nv
                + sum(self._enabled_directions)
            ),
            self._running_model_list,
            self._terminal_model,
        )
        self._problem.nthreads = ocp_params.n_threads
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

        self.init_debug_data_attributes()
        self._first_call = True

    @property
    def enabled_directions(self) -> int:
        return self._enabled_directions

    @property
    def frame_id(self) -> str:
        return self._data.running_model.differential.frame_id

    @property
    def oPc(self) -> npt.ArrayLike:
        return np.asarray(self._data.running_model.differential.oPc)


def get_globals():
    return globals()
