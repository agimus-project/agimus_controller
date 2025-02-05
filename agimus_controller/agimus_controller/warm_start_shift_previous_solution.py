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
import pinocchio
import crocoddyl

from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.warm_start_base import WarmStartBase


class WarmStartShiftPreviousSolution(WarmStartBase):
    """Generate a warm start by shifting in time the solution of the previous OCP iteration"""

    def __init__(self) -> None:
        super().__init__()

    def setup(self, rmodel: pinocchio.Model, timesteps: list[float]) -> None:
        """Build the action model to easily shift in time in `shift`.

        Args:
            rmodel (pinocchio.Model): The robot model
            timesteps (list[float]): list of time different between consecutive nodes of the OCP
                that produces the previous solution. It is required that:
                - timesteps[i] >= timesteps[0]
                - timesteps matches the OCP nodes time steps.

        """
        state = crocoddyl.StateMultibody(rmodel)
        actuation = crocoddyl.ActuationModelFull(state)
        differential = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state,
            actuation,
        )
        self._integrator = crocoddyl.IntegratedActionModelEuler(differential, 0.0)
        self._integrator_data = self._integrator.createData()
        self._timesteps = timesteps
        self._dt = self._timesteps[0]
        assert all(dt >= self._dt for dt in self._timesteps)

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
                self._integrator.dt = dt
                self._integrator.calc(self._integrator_data, xs[i], us[i])
                xs[i] = self._integrator_data.xnext
                # Keep the same control because we are still in the segment where
                # ocp.us[i] was to be applied.
                # TODO any better guess ? e.g.
                # - weighted average of us[i] and us[i+1] based on the time
                # - calculate us[i] so that xs[i+1] = f(xs[i], us[i])
