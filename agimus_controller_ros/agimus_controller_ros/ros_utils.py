import pinocchio as pin
import eigenpy
import numpy as np
import numpy.typing as npt
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_numpy_to_msg

import rclpy
from geometry_msgs.msg import Pose, Transform, Twist, Wrench
from agimus_msgs.msg import MpcInput, MpcDebug, Residual
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterValue

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


def array_to_ros_pose(pose_array: npt.NDArray[np.float64]) -> Pose:
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


def ros_twist_to_motion(twist: Twist) -> pin.Motion:
    """Convert geometry_msgs.msg.Twist to a pinocchio.Motion object"""
    return pin.Motion(
        np.array(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ]
        )
    )


def motion_to_ros_twist(motion: pin.Motion) -> Twist:
    """Convert pinocchio.Motion object to geometry_msgs.msg.Twist"""
    ros_twist = Twist()
    ros_twist.linear.x = motion.linear[0]
    ros_twist.linear.y = motion.linear[1]
    ros_twist.linear.z = motion.linear[2]
    ros_twist.angular.x = motion.angular[0]
    ros_twist.angular.y = motion.angular[1]
    ros_twist.angular.z = motion.angular[2]
    return ros_twist


def ros_wrench_to_force(wrench: Wrench) -> pin.Force:
    """Convert geometry_msgs.msg.Wrench to a pinocchio.Force object"""
    return pin.Force(
        np.array(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ]
        )
    )


def force_to_ros_wrench(force: pin.Force) -> Wrench:
    """Convert pinocchio.Force object to geometry_msgs.msg.Wrench"""
    ros_wrench = Wrench()
    ros_wrench.force.x = force.linear[0]
    ros_wrench.force.y = force.linear[1]
    ros_wrench.force.z = force.linear[2]
    ros_wrench.torque.x = force.angular[0]
    ros_wrench.torque.y = force.angular[1]
    ros_wrench.torque.z = force.angular[2]
    return ros_wrench


def transform_msg_to_se3(transform: Transform) -> pin.SE3:
    t = np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
    )
    q = eigenpy.Quaternion(
        transform.rotation.w,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
    )
    return pin.SE3(q, t)


def se3_to_transform_msg(M: pin.SE3) -> Transform:
    t = Transform()
    t.translation.x = M.translation[0]
    t.translation.y = M.translation[1]
    t.translation.z = M.translation[2]

    q = eigenpy.Quaternion(M.rotation)
    t.rotation.w = q.w
    t.rotation.x = q.x
    t.rotation.y = q.y
    t.rotation.z = q.z
    return t


def pose_msg_to_se3(pose: Pose) -> pin.SE3:
    t = np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]
    )
    q = eigenpy.Quaternion(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
    )
    return pin.SE3(q, t)


def se3_to_pose_msg(M: pin.SE3) -> Pose:
    t = Pose()
    t.position.x = M.translation[0]
    t.position.y = M.translation[1]
    t.position.z = M.translation[2]

    q = eigenpy.Quaternion(M.rotation)
    t.orientation.w = q.w
    t.orientation.x = q.x
    t.orientation.y = q.y
    t.orientation.z = q.z
    return t


def mpc_msg_to_weighted_traj_point(
    msg: MpcInput, time_ns: int
) -> WeightedTrajectoryPoint:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    xyz_quat_pose = ros_pose_to_array(msg.pose)
    twist = ros_twist_to_motion(msg.twist)
    force = ros_wrench_to_force(msg.force)
    traj_point = TrajectoryPoint(
        id=msg.id,
        time_ns=time_ns,
        robot_configuration=np.array(msg.q, dtype=np.float64),
        robot_velocity=np.array(msg.qdot, dtype=np.float64),
        robot_acceleration=np.array(msg.qddot, dtype=np.float64),
        robot_effort=np.array(msg.robot_effort, dtype=np.float64),
        end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat_pose)},
        end_effector_velocities={msg.ee_frame_name: twist},
        forces={msg.ee_frame_name: force},
    )

    traj_weights = TrajectoryPointWeights(
        w_robot_configuration=np.array(msg.w_q, dtype=np.float64),
        w_robot_velocity=np.array(msg.w_qdot, dtype=np.float64),
        w_robot_acceleration=np.array(msg.w_qddot, dtype=np.float64),
        w_robot_effort=np.array(msg.w_robot_effort, dtype=np.float64),
        w_end_effector_poses={msg.ee_frame_name: np.array(msg.w_pose)},
        w_end_effector_velocities={msg.ee_frame_name: np.array(msg.w_twist)},
        w_forces={msg.ee_frame_name: np.array(msg.w_force)},
    )

    return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)


def weighted_traj_point_to_mpc_msg(w_traj_point: WeightedTrajectoryPoint) -> MpcInput:
    """Build WeightedTrajectoryPoint object from MPCInput msg."""
    # get the first key of the dictionary

    ee_frame_name = next(iter(w_traj_point.point.end_effector_poses.keys()))

    msg = MpcInput()
    msg.id = w_traj_point.point.id
    msg.w_q = list(w_traj_point.weights.w_robot_configuration)
    msg.w_qdot = list(w_traj_point.weights.w_robot_velocity)
    msg.w_qddot = list(w_traj_point.weights.w_robot_acceleration)
    msg.w_robot_effort = list(w_traj_point.weights.w_robot_effort)

    w_poses = w_traj_point.weights.w_end_effector_poses
    if w_poses and len(w_poses) > 0:
        msg.w_pose = w_poses[ee_frame_name]

    w_vel = w_traj_point.weights.w_end_effector_velocities
    if w_vel and len(w_vel) > 0:
        msg.w_twist = w_vel[ee_frame_name]

    w_force = w_traj_point.weights.w_forces
    if w_force and len(w_force) > 0:
        msg.w_force = w_force[ee_frame_name]

    msg.q = list(w_traj_point.point.robot_configuration)
    msg.qdot = list(w_traj_point.point.robot_velocity)
    msg.qddot = list(w_traj_point.point.robot_acceleration)
    msg.robot_effort = list(w_traj_point.point.robot_effort)

    pose = w_traj_point.point.end_effector_poses
    if pose and len(pose) > 0:
        wMee = pose[ee_frame_name]
        if isinstance(wMee, pin.SE3):
            # This is the type defined in the annotation of the class definition
            # However, this is not enforced, hence the else.
            wMee = pin.SE3ToXYZQUAT(wMee)
        else:
            pass
        msg.pose = array_to_ros_pose(wMee)

    vel = w_traj_point.point.end_effector_velocities
    if vel and len(vel) > 0:
        msg.twist = motion_to_ros_twist(vel[ee_frame_name])

    force = w_traj_point.point.forces
    if force and len(force):
        msg.force = force_to_ros_wrench(force[ee_frame_name])

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

    mpc_debug_msg.trajectory_point_id = mpc_debug_data.reference_id
    # mpc_debug_msg.visual_servoing_is_active = (
    #    mpc_debug_data.ocp.visual_servoing_is_active
    # )
    mpc_debug_msg.kkt_norm = mpc_debug_data.ocp.kkt_norm
    mpc_debug_msg.nb_iter = mpc_debug_data.ocp.nb_iter
    mpc_debug_msg.nb_qp_iter = mpc_debug_data.ocp.nb_qp_iter
    return mpc_debug_msg


def get_params_from_node(
    requester_node: Node, target_node_name: str, target_params_name: list[str]
) -> list[ParameterValue]:
    """Returns parameters from a node"""
    service_name = f"/{target_node_name}/get_parameters"
    param_client = requester_node.create_client(GetParameters, service_name)
    while not param_client.wait_for_service(timeout_sec=1.0):
        requester_node.get_logger().info(
            f"Service {service_name} not available, waiting again..."
        )
    request = GetParameters.Request()
    request.names = target_params_name

    future = param_client.call_async(request)
    while not future.done():
        rclpy.spin_until_future_complete(requester_node, future, timeout_sec=1.0)
        requester_node.get_logger().info(
            f"Waiting for reply from service {service_name}..."
        )

    if future.result() is not None:
        # Ideally, values should be wrapped in rclpy.Parameter
        # using its from_parameter_msg method. This would change the
        # API so it is kept as is.
        return future.result().values
    else:
        raise ValueError(
            f"Failed to get parameter {target_params_name} from node {target_node_name}"
        )


def get_param_from_node(
    requester_node: Node, target_node_name: str, target_param_name: str
) -> ParameterValue:
    """Returns parameter from a node"""
    result = get_params_from_node(requester_node, target_node_name, [target_param_name])
    return result[0]
