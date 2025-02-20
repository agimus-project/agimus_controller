from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from pinocchio import SE3, Force
from agimus_controller.ocp_param_base import DTFactorsNSeq


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int | None = None
    robot_configuration: npt.NDArray[np.float64] | None = None
    robot_velocity: npt.NDArray[np.float64] | None = None
    robot_acceleration: npt.NDArray[np.float64] | None = None
    robot_effort: npt.NDArray[np.float64] | None = None
    forces: dict[Force] | None = None  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3] | None = None  # Dictionary of pinocchio.SE3

    @property
    def robot_state(self) -> npt.NDArray[np.float64]:
        return np.concatenate(self.robot_configuration, self.robot_velocity)

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

    @property
    def w_robot_state(self) -> npt.NDArray[np.float64]:
        return np.concatenate(self.w_robot_configuration, self.w_robot_velocity)

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
        assert self.horizon_indexes[-1] < len(self._buffer), (
            "Size of buffer must be at least horizon_indexes[-1]."
        )
        return [self._buffer[i] for i in self.horizon_indexes]

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]

    def __setitem__(self, index, value):
        self._buffer[index] = value
