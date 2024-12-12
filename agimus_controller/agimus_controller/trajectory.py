from collections import deque
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from pinocchio import SE3, Force


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int = None
    robot_configuration: npt.NDArray[np.float64] = None
    robot_velocity: npt.NDArray[np.float64] = None
    robot_acceleration: npt.NDArray[np.float64] = None
    robot_effort: npt.NDArray[np.float64] = None
    forces: dict[Force] = None  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3] = None  # Dictionary of pinocchio.SE3


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: npt.NDArray[np.float64] | None = None
    w_robot_velocity: npt.NDArray[np.float64] | None = None
    w_robot_acceleration: npt.NDArray[np.float64] | None = None
    w_robot_effort: npt.NDArray[np.float64] | None = None
    w_forces: dict[npt.NDArray[np.float64]] | None = None
    w_end_effector_poses: dict[npt.NDArray[np.float64]] | None = None


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weight: TrajectoryPointWeights


class TrajectoryBuffer(deque):
    """List of variable size in which the HPP trajectory nodes will be."""

    def clear_past(self, current_time_ns):
        while self and self[0].point.time_ns < current_time_ns:
            self.popleft()

    def horizon(self, horizon_size, dt_ocp):
        # TBD improve this implementation in case the dt_mpc != dt_ocp
        return self._buffer[: self._ocp.horizon_size]
