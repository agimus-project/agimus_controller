"""This module defines the WarmStartShiftPreviousSolution class.

The next warm start is generated based only on the previous solution. It
shifts the previous solution by the amount of the first time step.
- It supports non-constant time steps.
- It assumes the actuation model and the dynamics is the same all along the
  trajectory. To overcome this issue, it would need to take the models from
  the OCP.

When there is no previous solution, the warm start is calculated using an internal
WarmStartReference object.
"""

import numpy as np
import numpy.typing as npt
import crocoddyl

from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.ocp_param_base import OCPParamsBaseCroco


class WarmStartShiftPreviousSolution(WarmStartBase):
    """Generate a warm start by shifting in time the solution of the previous OCP iteration"""

    def __init__(self) -> None:
        super().__init__()

    def setup(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
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

        # Build the integrator
        state = crocoddyl.StateMultibody(robot_models.robot_model)
        actuation = crocoddyl.ActuationModelFull(state)
        cost_model = crocoddyl.CostModelSum(state)
        differential = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state,
            actuation,
            cost_model,
        )
        armature = robot_models.params.armature
        if armature.size > 0:
            differential.armature = armature
        self._integrator = crocoddyl.IntegratedActionModelEuler(differential, self._dt)
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
        x0 = np.concatenate(
            [initial_state.robot_configuration, initial_state.robot_velocity]
        )
        # TODO is copy needed ?
        xinit = self._previous_solution.states[1:].copy()
        uinit = self._previous_solution.feed_forward_terms.copy()
        return x0, xinit, uinit

    def shift(self):
        """Shift the previous solution by self._dt by applying the forward dynamics."""
        xs = self._previous_solution.states
        us = self._previous_solution.feed_forward_terms

        nb_timesteps = len(self._timesteps)
        assert len(xs) == nb_timesteps + 1
        assert len(us) == nb_timesteps
        for i, dt in enumerate(self._timesteps):
            if dt == self._dt:
                xs[i] = xs[i + 1]
                # for the last running model, i+1 is the terminal model.
                # There is no control for this one. The result of the current loop is
                # that if two last control will be equal.
                if i < nb_timesteps - 1:
                    us[i] = us[i + 1]
            else:
                assert dt > self._dt
                self._integrator.calc(self._integrator_data, xs[i], us[i])
                xs[i] = self._integrator_data.xnext.copy()
                # Keep the same control because we are still in the segment where
                # ocp.us[i] was to be applied.
                # TODO any better guess ? e.g.
                # - weighted average of us[i] and us[i+1] based on the time
                # - calculate us[i] so that xs[i+1] = f(xs[i], us[i])
