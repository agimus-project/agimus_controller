from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from pinocchio import Force, Motion, SE3
from agimus_controller.ocp_param_base import DTFactorsNSeq


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    id: int | None = None
    time_ns: int | None = None
    robot_configuration: npt.NDArray[np.float64] | None = None
    robot_velocity: npt.NDArray[np.float64] | None = None
    robot_acceleration: npt.NDArray[np.float64] | None = None
    robot_effort: npt.NDArray[np.float64] | None = None
    forces: dict[Force] | None = None  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3] | None = None  # Dictionary of pinocchio.SE3
    end_effector_velocities: dict[Motion] | None = (
        None  # Dictionary of pinocchio.Motion
    )

    @property
    def robot_state(self) -> npt.NDArray[np.float64]:
        return np.concatenate((self.robot_configuration, self.robot_velocity))

    def __eq__(self, other):
        if not isinstance(other, TrajectoryPoint):
            return False

        # Compare scalar values directly
        if self.time_ns != other.time_ns:
            return False

        # Compare numpy arrays (ignoring None values)
        if (
            self.robot_configuration is not None
            and other.robot_configuration is not None
        ):
            if not np.array_equal(self.robot_configuration, other.robot_configuration):
                return False
        elif (
            self.robot_configuration is not None
            or other.robot_configuration is not None
        ):
            return False

        if self.robot_velocity is not None and other.robot_velocity is not None:
            if not np.array_equal(self.robot_velocity, other.robot_velocity):
                return False
        elif self.robot_velocity is not None or other.robot_velocity is not None:
            return False

        if self.robot_acceleration is not None and other.robot_acceleration is not None:
            if not np.array_equal(self.robot_acceleration, other.robot_acceleration):
                return False
        elif (
            self.robot_acceleration is not None or other.robot_acceleration is not None
        ):
            return False

        if self.robot_effort is not None and other.robot_effort is not None:
            if not np.array_equal(self.robot_effort, other.robot_effort):
                return False
        elif self.robot_effort is not None or other.robot_effort is not None:
            return False

        # Compare dictionaries (forces and end_effector_poses)
        if self.forces != other.forces:
            return False

        if self.end_effector_poses != other.end_effector_poses:
            return False

        if self.end_effector_velocities != other.end_effector_velocities:
            return False

        return True


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: npt.NDArray[np.float64] | None = None
    w_robot_velocity: npt.NDArray[np.float64] | None = None
    w_robot_acceleration: npt.NDArray[np.float64] | None = None
    w_robot_effort: npt.NDArray[np.float64] | None = None
    w_forces: dict[npt.NDArray[np.float64]] | None = None
    w_end_effector_poses: dict[npt.NDArray[np.float64]] | None = None
    w_end_effector_velocities: dict[npt.NDArray[np.float64]] | None = None
    w_collision_avoidance: np.float64 | None = None

    @property
    def w_robot_state(self) -> npt.NDArray[np.float64]:
        return np.concatenate((self.w_robot_configuration, self.w_robot_velocity))

    def __eq__(self, other):
        if not isinstance(other, TrajectoryPointWeights):
            return False

        # Compare numpy arrays (weights)
        if (
            self.w_robot_configuration is not None
            and other.w_robot_configuration is not None
        ):
            if not np.array_equal(
                self.w_robot_configuration, other.w_robot_configuration
            ):
                return False
        elif (
            self.w_robot_configuration is not None
            or other.w_robot_configuration is not None
        ):
            return False

        if self.w_robot_velocity is not None and other.w_robot_velocity is not None:
            if not np.array_equal(self.w_robot_velocity, other.w_robot_velocity):
                return False
        elif self.w_robot_velocity is not None or other.w_robot_velocity is not None:
            return False

        if (
            self.w_robot_acceleration is not None
            and other.w_robot_acceleration is not None
        ):
            if not np.array_equal(
                self.w_robot_acceleration, other.w_robot_acceleration
            ):
                return False
        elif (
            self.w_robot_acceleration is not None
            or other.w_robot_acceleration is not None
        ):
            return False

        if self.w_robot_effort is not None and other.w_robot_effort is not None:
            if not np.array_equal(self.w_robot_effort, other.w_robot_effort):
                return False
        elif self.w_robot_effort is not None or other.w_robot_effort is not None:
            return False

        if self.w_forces != other.w_forces:
            return False

        if self.w_end_effector_poses != other.w_end_effector_poses:
            return False

        if self.w_end_effector_velocities != other.w_end_effector_velocities:
            return False

        if self.w_collision_avoidance != other.w_collision_avoidance:
            return False

        return True


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weights: TrajectoryPointWeights

    def __eq__(self, other):
        if not isinstance(other, WeightedTrajectoryPoint):
            return False

        # Compare the 'point' and 'weight' attributes
        if self.point != other.point:
            return False

        if self.weights != other.weights:
            return False

        return True


class TrajectoryBuffer(object):
    """List of variable size in which the HPP trajectory nodes will be."""

    def __init__(self, dt_factor_n_seq: DTFactorsNSeq):
        self._buffer = []
        self.dt_factor_n_seq = deepcopy(dt_factor_n_seq)
        self.horizon_indexes = self.compute_horizon_indexes()

    def append(self, item):
        self._buffer.append(item)

    def pop(self, index=-1):
        return self._buffer.pop(index)

    def clear_past(self):
        if self._buffer:
            self._buffer.pop(0)

    def compute_horizon_indexes(self):
        n_states = sum(sn for sn in self.dt_factor_n_seq.n_steps) + 1
        indexes = [0] * n_states
        i = 1
        for factor, sn in zip(
            self.dt_factor_n_seq.factors, self.dt_factor_n_seq.n_steps
        ):
            for _ in range(sn):
                indexes[i] = factor + indexes[i - 1]
                i += 1

        assert i == len(indexes)
        assert indexes[0] == 0, "First time step must be 0"
        assert all(t0 <= t1 for t0, t1 in zip(indexes[:-1], indexes[1:])), (
            "Time steps must be increasing"
        )
        return indexes

    @property
    def horizon(self):
        last = self._buffer[-1]
        return [
            self._buffer[i] if i < len(self._buffer) else last
            for i in self.horizon_indexes
        ]

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]

    def __setitem__(self, index, value):
        self._buffer[index] = value


def interpolate_weights(
    p1: TrajectoryPointWeights, p2: TrajectoryPointWeights, alpha: float
) -> TrajectoryPointWeights:
    """Linearly interpolates weights between two trajectory points.

    Args:
        p1 (TrajectoryPointWeights): Starting weights for the interpolation.
        p2 (TrajectoryPointWeights): End weights for the interpolation.
        alpha (float): Interpolation coefficient [0.0, 1.0].

    Returns:
        TrajectoryPointWeights: Interpolated point.
    """
    alpha = np.clip(alpha, 0.0, 1.0)

    def _w_interpolate(
        w1: np.float64 | npt.ArrayLike, w2: np.float64 | npt.ArrayLike
    ) -> np.float64 | npt.ArrayLike:
        return (1.0 - alpha) * w1 + alpha * w2

    def _interpolate_dict(ee_w_1: dict, ee_w_2: dict) -> npt.ArrayLike:
        res = {}
        for frame in set(ee_w_1.keys()) | set(ee_w_2.keys()):
            if frame not in ee_w_1:
                res[frame] = _w_interpolate(ee_w_1[frame], np.zeros_like(ee_w_1[frame]))
            elif frame not in ee_w_2:
                res[frame] = _w_interpolate(np.zeros_like(ee_w_2[frame]), ee_w_2[frame])
            else:
                res[frame] = _w_interpolate(ee_w_1[frame], ee_w_2[frame])
        return res

    def _interpolate_args(
        w_1: np.float64 | npt.ArrayLike | dict,
        w_2: np.float64 | npt.ArrayLike | dict,
        arg,
    ) -> npt.ArrayLike:
        if isinstance(w_1, dict):
            return _interpolate_dict(w_1, w_2)
        return _w_interpolate(w_1, w_2)

    return TrajectoryPointWeights(
        **{
            arg: _interpolate_args(getattr(p1, arg), getattr(p2, arg), arg)
            for arg in TrajectoryPointWeights.__match_args__
        }
    )
