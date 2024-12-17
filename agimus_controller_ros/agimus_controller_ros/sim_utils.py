import pinocchio as pin
import numpy as np
import numpy.typing as npt


from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from agimus_msgs.msg import MpcInput

from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


def ros_pose_to_array(pose: Pose) -> npt.NDArray[np.float64]:
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


def convert_float_to_ros_duration_msg(time: float) -> Duration:
    """Convert float duration to ROS duration msg."""
    ros_duration = Duration()
    ros_duration.sec = int(time)
    ros_duration.nanosec = int((time % 1) * 1e9)
    return ros_duration


def mpc_msg_to_weighted_traj_point(msg: MpcInput) -> WeightedTrajectoryPoint:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    xyz_quat_pose = ros_pose_to_array(msg.pose)
    traj_point = TrajectoryPoint(
        robot_configuration=msg.q,
        robot_velocity=msg.qdot,
        robot_acceleration=msg.qddot,
        end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat_pose)},
    )

    traj_weights = TrajectoryPointWeights(
        w_robot_configuration=msg.q_w,
        w_robot_velocity=msg.qdot_w,
        w_robot_acceleration=msg.qddot_w,
        w_end_effector_poses=msg.pose_w,
    )

    return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
