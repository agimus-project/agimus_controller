import numpy as np
from collections import deque
from dataclasses import dataclass
from geometry_msgs.msg import Pose
from pinocchio import SE3, Force


def ros_pose_to_array(pose: Pose):
    """Convert geometry_msgs.msg.Pose to a 7d numpy array"""
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )


@dataclass
class TrajectoryPoint:
    """Trajectory point aiming at being a reference for the MPC."""

    time_ns: int = None
    robot_configuration: np.ndarray = None
    robot_velocity: np.ndarray = None
    robot_acceleration: np.ndarray = None
    robot_effort: np.ndarray = None
    forces: dict[Force] = None  # Dictionary of pinocchio.Force
    end_effector_poses: dict[SE3] = None  # Dictionary of pinocchio.SE3


@dataclass
class TrajectoryPointWeights:
    """Trajectory point weights aiming at being set in the MPC costs."""

    w_robot_configuration: np.ndarray = None
    w_robot_velocity: np.ndarray = None
    w_robot_acceleration: np.ndarray = None
    w_robot_effort: np.ndarray = None
    w_forces: dict[np.ndarray] = None
    w_end_effector_poses: dict[np.ndarray] = None


@dataclass
class WeightedTrajectoryPoint:
    """Trajectory point and it's corresponding weights."""

    point: TrajectoryPoint
    weights: TrajectoryPointWeights


class TrajectoryBuffer(deque):
    """List of variable size in which the HPP trajectory nodes will be."""

    def clear_past(self, current_time_ns):
        while self and self[0].point.time_ns < current_time_ns:
            self.popleft()
