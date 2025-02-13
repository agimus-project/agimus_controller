import pinocchio as pin
import numpy as np
import numpy.typing as npt
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_numpy_to_msg

from geometry_msgs.msg import Pose
from agimus_msgs.msg import MpcInput, MpcDebug

from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)
from agimus_controller.mpc_data import OCPResults, MPCDebugData


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


def mpc_debug_data_to_msg(
    ocp_res: OCPResults, mpc_debug_data: MPCDebugData
) -> MpcDebug:
    """Build MPC debug data message."""
    mpc_debug_msg = MpcDebug()
    mpc_debug_msg.states_predictions = matrix_numpy_to_msg(np.array(ocp_res.states))
    mpc_debug_msg.control_predictions = matrix_numpy_to_msg(
        np.array(ocp_res.feed_forward_terms)
    )
    if mpc_debug_data.ocp.collision_distance_residuals is not None:
        mpc_debug_msg.collision_distance_residuals = matrix_numpy_to_msg(
            np.array(mpc_debug_data.ocp.collision_distance_residuals)
        )
    else:
        mpc_debug_msg.collision_distance_residuals = matrix_numpy_to_msg(np.zeros((1)))

    mpc_debug_msg.kkt_norm = mpc_debug_data.ocp.kkt_norm
    mpc_debug_msg.nb_iter = mpc_debug_data.ocp.nb_iter
    mpc_debug_msg.nb_qp_iter = mpc_debug_data.ocp.nb_qp_iter
    return mpc_debug_msg
