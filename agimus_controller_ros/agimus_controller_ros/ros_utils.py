import pinocchio as pin
import numpy as np
import numpy.typing as npt
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_numpy_to_msg

from geometry_msgs.msg import Pose
from agimus_msgs.msg import MpcInput, MpcDebug, Residual

from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.mpc_data import MPCDebugData


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


def array_to_ros_pose(pose_array: Pose) -> npt.NDArray[np.float64]:
    """Convert geometry_msgs.msg.Pose to a 7d numpy array"""
    ros_pose = Pose()
    ros_pose.position.x = pose_array[0]
    ros_pose.position.y = pose_array[1]
    ros_pose.position.z = pose_array[2]
    ros_pose.orientation.x = pose_array[3]
    ros_pose.orientation.y = pose_array[4]
    ros_pose.orientation.z = pose_array[5]
    ros_pose.orientation.w = pose_array[6]
    return ros_pose


def mpc_msg_to_weighted_traj_point(
    msg: MpcInput, time_ns: int
) -> WeightedTrajectoryPoint:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    xyz_quat_pose = ros_pose_to_array(msg.pose)
    traj_point = TrajectoryPoint(
        time_ns=time_ns,
        robot_configuration=np.array(msg.q, dtype=np.float64),
        robot_velocity=np.array(msg.qdot, dtype=np.float64),
        robot_acceleration=np.array(msg.qddot, dtype=np.float64),
        robot_effort=np.array(msg.robot_effort, dtype=np.float64),
        end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat_pose)},
    )

    traj_weights = TrajectoryPointWeights(
        w_robot_configuration=np.array(msg.w_q, dtype=np.float64),
        w_robot_velocity=np.array(msg.w_qdot, dtype=np.float64),
        w_robot_acceleration=np.array(msg.w_qddot, dtype=np.float64),
        w_robot_effort=np.array(msg.w_robot_effort, dtype=np.float64),
        w_end_effector_poses={msg.ee_frame_name: msg.w_pose},
    )

    return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)


def weighted_traj_point_to_mpc_msg(w_traj_point: WeightedTrajectoryPoint) -> MpcInput:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    # get the first key of the dictionary
    ee_frame_name = next(iter(w_traj_point.point.end_effector_poses.keys()))

    msg = MpcInput()
    msg.w_q = w_traj_point.weights.w_robot_configuration
    msg.w_qdot = w_traj_point.weights.w_robot_velocity
    msg.w_qddot = w_traj_point.weights.w_robot_acceleration
    msg.w_robot_effort = w_traj_point.weights.w_robot_effort
    msg.w_pose = next(iter(w_traj_point.weights.w_end_effector_poses.values()))
    msg.q = list(w_traj_point.point.robot_configuration)
    msg.qdot = list(w_traj_point.point.robot_velocity)
    msg.qddot = list(w_traj_point.point.robot_acceleration)
    msg.robot_effort = list(w_traj_point.point.robot_effort)
    msg.pose = array_to_ros_pose(w_traj_point.point.end_effector_poses[ee_frame_name])
    msg.ee_frame_name = ee_frame_name
    return msg


def mpc_debug_data_to_msg(mpc_debug_data: MPCDebugData) -> MpcDebug:
    """Build MPC debug data message."""
    mpc_debug_msg = MpcDebug()
    mpc_debug_msg.states_predictions = matrix_numpy_to_msg(
        np.array(mpc_debug_data.ocp.result.states)
    )
    mpc_debug_msg.control_predictions = matrix_numpy_to_msg(
        np.array(mpc_debug_data.ocp.result.feed_forward_terms)
    )
    for name, data in mpc_debug_data.ocp.residuals:
        mpc_debug_msg.residuals.append(
            Residual(name=name, data=matrix_numpy_to_msg(np.asarray(data)))
        )
    for name, data in mpc_debug_data.ocp.references:
        mpc_debug_msg.references.append(
            Residual(name=name, data=matrix_numpy_to_msg(np.asarray(data)))
        )

    mpc_debug_msg.kkt_norm = mpc_debug_data.ocp.kkt_norm
    mpc_debug_msg.nb_iter = mpc_debug_data.ocp.nb_iter
    mpc_debug_msg.nb_qp_iter = mpc_debug_data.ocp.nb_qp_iter
    return mpc_debug_msg
