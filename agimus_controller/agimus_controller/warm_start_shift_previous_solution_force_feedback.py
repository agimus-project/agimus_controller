"""This module defines the WarmStartShiftPreviousSolutionForceFeedback class.

The next warm start is generated based only on the previous solution. It
shifts the previous solution by the amount of the first time step.
- It supports non-constant time steps.
- It assumes the actuation model and the dynamics is the same all along the
  trajectory. To overcome this issue, it would need to take the models from
  the OCP.

When there is no previous solution, the warm start is calculated using an internal
WarmStartReference object.
"""

import typing as T

import crocoddyl
import numpy as np
import numpy.typing as npt
import resource_retriever as r
import yaml
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.ocp.ocp_croco_generic import BuildData, ShootingProblem
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.warm_start_shift_previous_solution import (
    WarmStartShiftPreviousSolution,
)


class WarmStartShiftPreviousSolutionForceFeedback(WarmStartShiftPreviousSolution):
    def __init__(self) -> None:
        super().__init__()

    def setup(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
        yaml_file: T.Union[str, T.IO],
    ) -> None:
        """Build the action model to easily shift in time in `shift`.

        Args:
            rmodel (pinocchio.Model): The robot model
            timesteps (list[float]): list of time different between consecutive nodes of the OCP
                that produces the previous solution. It is required that:
                - timesteps[i] >= timesteps[0]
                - timesteps matches the OCP nodes time steps.
        """
        self._timesteps = ocp_params.timesteps
        self._dt = self._timesteps[0]
        assert ocp_params.dt == self._timesteps[0]
        assert all(dt >= self._dt for dt in self._timesteps)

        data = yaml.safe_load(r.get(yaml_file))
        data["running_model"]["differential"]["costs"] = []

        self._state = crocoddyl.StateMultibody(robot_models.robot_model)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        model = ShootingProblem(**data).running_model
        self._enabled_directions = model.differential.enabled_directions
        self._integrator = model.build(
            BuildData(self._state, self._actuation, robot_models.collision_model)
        )
        self._integrator.differential.armature = robot_models.armature
        self._integrator.dt = self._dt

        self._integrator_data = self._integrator.createData()

    def generate(
        self,
        initial_state: TrajectoryPoint,
        reference_trajectory: list[TrajectoryPoint],
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
    ]:
        assert self._previous_solution is not None, (
            "WarmStartBase.update_previous_solution should have been called before generate can work."
        )
        self.shift()
        if sum(self._enabled_directions) > 1:
            force = initial_state.forces.linear
        else:
            force = initial_state.forces.linear[self._enabled_directions]

        x0 = np.concatenate(
            [
                initial_state.robot_configuration,
                initial_state.robot_velocity,
                force,
            ]
        )
        # TODO is copy needed ?
        xinit = self._previous_solution.states.copy()
        uinit = self._previous_solution.feed_forward_terms.copy()
        return x0, xinit, uinit
