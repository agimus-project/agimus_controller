import time
from agimus_controller.ocp_base import OCPBase
from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    WeightedTrajectoryPoint,
    TrajectoryBuffer,
)
from agimus_controller.mpc_data import OCPResults, MPCDebugData


class MPC(object):
    def __init__(self) -> None:
        self._ocp = None
        self._warm_start = None
        self._mpc_debug_data: MPCDebugData = None
        self._buffer = TrajectoryBuffer()

    def setup(self, ocp: OCPBase, warm_start: WarmStartBase) -> None:
        self._ocp = ocp
        self._warm_start = warm_start

    def run(self, initial_state: TrajectoryPoint, current_time_ns: int) -> OCPResults:
        assert self._ocp is not None
        assert self._warm_start is not None
        timer1 = time.perf_counter_ns()
        self._buffer.clear_past(current_time_ns)
        reference_trajectory = self._extract_horizon_from_buffer()
        self._ocp.set_reference_horizon(reference_trajectory)
        timer2 = time.perf_counter_ns()
        x0, x_init, u_init = self._warm_start.generate(
            initial_state, reference_trajectory, self._ocp.debug_data.result
        )
        timer3 = time.perf_counter_ns()
        self._ocp.solve(x0, x_init, u_init)
        timer4 = time.perf_counter_ns()

        # Extract the solution.
        self._mpc_debug_data = self._ocp.debug_data
        self._mpc_debug_data.duration_iteration_ns = timer4 - timer1
        self._mpc_debug_data.duration_horizon_update_ns = timer2 - timer1
        self._mpc_debug_data.duration_generate_warm_start_ns = timer3 - timer2
        self._mpc_debug_data.duration_ocp_solve_ns = timer4 - timer3

        return self._ocp.ocp_results

    @property
    def mpc_debug_data(self) -> MPCDebugData:
        return self._mpc_debug_data

    def add_trajectory_point(self, trajectory_point: WeightedTrajectoryPoint):
        self._buffer.append(trajectory_point)

    def add_trajectory_points(self, trajectory_points: list[WeightedTrajectoryPoint]):
        self._buffer.extend(trajectory_points)

    def _extract_horizon_from_buffer(self):
        return self._buffer[: self._ocp.horizon_size]
