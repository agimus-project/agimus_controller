import time
import numpy.typing as npt

from agimus_controller.mpc_data import OCPResults, MPCDebugData
from agimus_controller.ocp_base import OCPBase
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    WeightedTrajectoryPoint,
)
from agimus_controller.warm_start_base import WarmStartBase


class MPC(object):
    def __init__(self) -> None:
        self._ocp = None
        self._warm_start = None
        self._mpc_debug_data: MPCDebugData = None
        self._buffer = None

    def setup(
        self,
        ocp: OCPBase,
        warm_start: WarmStartBase,
        buffer: TrajectoryBuffer,
    ) -> None:
        self._ocp = ocp
        self._warm_start = warm_start
        self._buffer = buffer
        self._mpc_debug_data = MPCDebugData(ocp=self._ocp.debug_data)

    def run(self, initial_state: TrajectoryPoint, current_time_ns: int) -> OCPResults:
        assert self._ocp is not None
        assert self._warm_start is not None
        timer1 = time.perf_counter_ns()

        # Ensure that you have enough data in the buffer.
        if len(self._buffer) < self._ocp.n_controls + 1:
            return None
        reference_trajectory = self._extract_horizon_from_buffer()
        self._ocp.set_reference_weighted_trajectory(reference_trajectory)
        timer2 = time.perf_counter_ns()

        # TODO avoid building this list by making warm start classes use a reference trajectory with weights.
        reference_trajectory_points = [el.point for el in reference_trajectory]
        x0, x_init, u_init = self._warm_start.generate(
            initial_state, reference_trajectory_points
        )
        assert len(x_init) == self._ocp.n_controls + 1
        assert len(u_init) == self._ocp.n_controls

        timer3 = time.perf_counter_ns()
        self._ocp.solve(x0, x_init, u_init)
        self._warm_start.update_previous_solution(self._ocp.ocp_results)
        self._buffer.clear_past()
        timer4 = time.perf_counter_ns()

        # Extract the solution.
        self._mpc_debug_data.ocp = self._ocp.debug_data
        self._mpc_debug_data.duration_iteration_ns = timer4 - timer1
        self._mpc_debug_data.duration_horizon_update_ns = timer2 - timer1
        self._mpc_debug_data.duration_generate_warm_start_ns = timer3 - timer2
        self._mpc_debug_data.duration_ocp_solve_ns = timer4 - timer3

        return self._ocp.ocp_results

    def integrate(
        self, state: TrajectoryPoint, control: npt.NDArray
    ) -> TrajectoryPoint:
        """Integrate the control starting from state during duration dt.

        Returns:
            the same TrajectoryPoint object, where robot_configuration and robot_velocity have been modified.
        """
        x = self._ocp.integrate(state.robot_state, control)
        state.time_ns += int(self.ocp.dt * 1e-9)
        state.robot_configuration = x[: len(state.robot_configuration)]
        state.robot_velocity = x[len(state.robot_configuration) :]
        return state

    @property
    def mpc_debug_data(self) -> MPCDebugData:
        return self._mpc_debug_data

    def append_trajectory_point(self, trajectory_point: WeightedTrajectoryPoint):
        self._buffer.append(trajectory_point)

    def append_trajectory_points(
        self, trajectory_points: list[WeightedTrajectoryPoint]
    ):
        self._buffer.extend(trajectory_points)

    def _extract_horizon_from_buffer(self):
        return self._buffer.horizon
