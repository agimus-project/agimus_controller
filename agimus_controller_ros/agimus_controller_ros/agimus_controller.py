#!/usr/bin/env python3
import numpy as np
import time
import os
import resource_retriever
from functools import partial

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import rclpy.time
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Pose, Twist
from agimus_msgs.msg import MpcInput, MpcDebug
from rcl_interfaces.srv import GetParameters
import builtin_interfaces

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import linear_feedback_controller_msgs_py.lfc_py_types as lfc_py_types
from linear_feedback_controller_msgs_py.numpy_conversions import (
    sensor_msg_to_numpy,
    control_numpy_to_msg,
)
from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState

from agimus_controller.mpc import MPC
from agimus_controller.mpc_data import OCPResults
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.warm_start_shift_previous_solution import (
    WarmStartShiftPreviousSolution,
)
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters


from agimus_controller_ros.ros_utils import (
    mpc_msg_to_weighted_traj_point,
    mpc_debug_data_to_msg,
    transform_msg_to_se3,
    ros_pose_to_se3,
    sensor_ff_to_planar_state,
)


from agimus_controller.trajectory import TrajectoryBuffer, TrajectoryPoint
from agimus_controller_ros.agimus_controller_parameters import agimus_controller_params


class RobotModelsMixin:
    def init_ros_robot_creation(self) -> None:
        self.q0 = None
        self.robot_description_msg = None
        self.environment_msg = None
        self.robot_srdf_description_msg = None

        # moving_joint_names is fetched asynchronously in initialization_callback
        # to avoid blocking __init__ when the LFC service is not yet available.
        self.moving_joint_names = None
        self._lfc_param_future = None
        self._lfc_param_client = self.create_client(
            GetParameters, "/linear_feedback_controller/get_parameters"
        )

        self.subscriber_robot_description = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.subscriber_environment_description = self.create_subscription(
            String,
            "/environment_description",
            self.environment_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.subscriber_robot_srdf_description = self.create_subscription(
            String,
            "/robot_srdf_description",
            self.robot_srdf_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.state_subscriber = self.create_subscription(
            JointState,
            "joint_states",
            self.joint_states_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

    def robot_description_callback(self, msg: String) -> None:
        """Set robot description xml msg."""
        self.robot_description_msg = msg

    def environment_description_callback(self, msg: String) -> None:
        """Set environment description xml msg."""
        self.environment_msg = msg

    def robot_srdf_description_callback(self, msg: String) -> None:
        """Set robot srdf description xml msg."""
        self.robot_srdf_description_msg = msg

    def joint_states_callback(self, joint_states_msg: JointState) -> None:
        """Set joint state reference."""
        if (
            self.robot_description_msg is None
            or self.environment_msg is None
            or self.robot_srdf_description_msg is None
        ):
            return
        if self.q0 is None:
            self.q0 = np.array(joint_states_msg.position)
            self.destroy_subscription(self.state_subscriber)
            self.destroy_subscription(self.subscriber_robot_description)
            self.destroy_subscription(self.subscriber_robot_srdf_description)
            self.destroy_subscription(self.subscriber_environment_description)

    def ros_robot_ready(self) -> bool:
        if self.robot_description_msg is None:
            self.get_logger().warn(
                "Waiting for robot descriptions...",
                throttle_duration_sec=5.0,
            )
        if self.environment_msg is None:
            self.get_logger().warn(
                "Waiting for environment description...",
                throttle_duration_sec=5.0,
            )
        if self.robot_srdf_description_msg is None:
            self.get_logger().warn(
                "Waiting for robot SRDF description...",
                throttle_duration_sec=5.0,
            )
        return self.q0 is not None

    def create_robot_models(self, **robot_model_parameters_kwargs) -> None:
        robot_params = RobotModelParameters(
            robot_urdf=self.robot_description_msg.data,
            env_urdf=self.environment_msg.data,
            srdf=self.robot_srdf_description_msg.data,
            moving_joint_names=self.moving_joint_names,
            **robot_model_parameters_kwargs,
        )
        self.robot_models = RobotModels(robot_params)
        self.rmodel = self.robot_models._robot_model

        self.get_logger().info("Robot Models initialized")


class AgimusController(Node, RobotModelsMixin):
    """Agimus controller's ROS 2 node class."""

    def __init__(self, node_name: str = "agimus_controller_node") -> None:
        """Get ROS parameters, initialize trajectory buffer and ros attributes."""
        super().__init__(node_name)
        self.param_listener = agimus_controller_params.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.params.ocp.armature = np.array(self.params.ocp.armature)
        self.traj_buffer = TrajectoryBuffer(self.params.ocp.dt_factor_n_seq)
        self.params.collision_pairs = [
            (
                self.params.get_entry(collision_pair_name).first,
                self.params.get_entry(collision_pair_name).second,
            )
            for collision_pair_name in self.params.collision_pairs_names
        ]
        # Check that the number of threads is suitable
        # Source: https://stackoverflow.com/a/55423170
        if self.params.ocp.n_threads > len(os.sched_getaffinity(0)):
            self.get_logger().warn(
                f"The requested number of threads {self.params.ocp.n_threads} is higher than the number of usable cores {len(os.sched_getaffinity(0))}"
            )
        self.ocp_params = OCPParamsBaseCroco(
            dt=self.params.ocp.dt,
            dt_factor_n_seq=self.params.ocp.dt_factor_n_seq,
            horizon_size=self.params.ocp.horizon_size,
            solver_iters=self.params.ocp.max_iter,
            callbacks=self.params.ocp.activate_callback,
            qp_iters=self.params.ocp.max_qp_iter,
            use_debug_data=self.params.publish_debug_data,
            n_threads=self.params.ocp.n_threads,
            max_solve_time=self.params.ocp.max_solve_time,
            termination_tolerance=self.params.ocp.termination_tolerance,
        )
        self.last_point = None
        self.first_run_done = False
        self.rmodel = None
        self.mpc = None
        self.np_sensor_msg = None
        # Stores the OCP result to be able to publish it
        # at next iteration, when using a constant delay
        self._ocp_res = None

        self.initialize_ros_attributes()
        self.get_logger().info(
            f"Init done. Horizon total time is {self.ocp_params.total_time}"
        )

    def initialize_ros_attributes(self) -> None:
        """Initialize ROS related attributes such as Publishers, Subscribers and Timers"""
        self.sensor_msg = None
        self.control_msg = None

        self.init_ros_robot_creation()

        self.sensor_subscriber = self.create_subscription(
            Sensor,
            "sensor",
            self.sensor_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.subscriber_mpc_input = self.create_subscription(
            MpcInput,
            "mpc_input",
            self.mpc_input_callback,
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        # Create topic subscribers for MPC geometries
        self.geom_subscribers = []
        for geom_name in self.params.moving_geometries_names:
            # Skip invalid topic name
            if geom_name == "":
                continue
            self.geom_subscribers.append(
                self.create_subscription(
                    Pose,
                    geom_name,
                    partial(self.update_geom_pose_callback, geom_name=geom_name),
                    10,
                )
            )

        self.control_publisher = self.create_publisher(
            Control,
            "control",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.base_cmd_vel_publisher = self.create_publisher(
            Twist,
            "base_cmd_vel",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        if self.params.publish_buffer_size:
            self.ocp_buffer_size_pub = self.create_publisher(
                Int32,
                "mpc_buffer_size",
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                ),
            )

        if self.params.publish_debug_data:
            self.ocp_solve_time_pub = self.create_publisher(
                builtin_interfaces.msg.Duration,
                "ocp_solve_time",
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                ),
            )
            self.ocp_x0_pub = self.create_publisher(
                Sensor,
                "ocp_x0",
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                ),
            )
            self.mpc_debug_pub = self.create_publisher(
                MpcDebug,
                "mpc_debug",
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                ),
            )

        self._init_timer = self.create_timer(0.1, self.initialization_callback)

    def initialize_tf_listener(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def update_transforms(self):
        latest_time = rclpy.time.Time()
        current_time = self.get_clock().now()
        transforms = self.ocp.input_transforms
        for key in transforms:
            parent_frame, child_frame = key
            try:
                # Note that `latest_time` is used and not `current_time` because
                # the latter leads to an interpolation into the future.
                # see https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Learning-About-Tf2-And-Time-Cpp.html#update-the-listener-node
                t = self.tf_buffer.lookup_transform(
                    parent_frame, child_frame, latest_time
                )
                t_stamp = rclpy.time.Time.from_msg(t.header.stamp)
                # if timestamp is 0, we assume it's a static transform published once at the beginning
                if (
                    t_stamp.nanoseconds != 0
                    and (current_time - t_stamp).nanoseconds > 0.5e9
                ):
                    self.get_logger().info(
                        f"Transform {parent_frame} to {child_frame} is too old. Latest time is {t.header.stamp}",
                        throttle_duration_sec=1.0,
                    )
                    M = None
                else:
                    M = transform_msg_to_se3(t.transform)
            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {parent_frame} to {child_frame}: {ex}",
                    throttle_duration_sec=1.0,
                )
                M = None
            transforms[key] = M

    def setup_mpc(self):
        """Creates mpc, ocp, warmstart"""
        yaml_file = self.params.ocp.definition_yaml_file
        if yaml_file == "":
            yaml_file = OCPCrocoGeneric.get_default_yaml_file("ocp_goal_reaching.yaml")
        else:
            yaml_file = resource_retriever.get_filename(yaml_file, use_protocol=False)
        self.get_logger().info(f"Loading OCP definition file {yaml_file}")
        self.ocp = OCPCrocoGeneric(self.robot_models, self.ocp_params, yaml_file)

        if len(self.ocp.input_transforms) > 0:
            self.initialize_tf_listener()

        self._ws_shift = WarmStartShiftPreviousSolution()
        self._ws_shift.setup(self.robot_models, self.ocp_params)

        self.mpc = MPC()
        self.mpc.setup(ocp=self.ocp, warm_start=self._ws_shift, buffer=self.traj_buffer)

    def setup_mpc_initial_guess(self):
        # Use WarmStartReference for initialization
        ws_ref = WarmStartReference()
        ws_ref.setup(self.robot_models._robot_model)

        np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)
        q_arm = np_sensor_msg.joint_state.position
        v_arm = np_sensor_msg.joint_state.velocity
        if self.params.planar_base:
            q_sensor, v_sensor = sensor_ff_to_planar_state(
                self.sensor_msg, q_arm, v_arm
            )
        else:
            q_sensor, v_sensor = q_arm, v_arm
        initial_state = TrajectoryPoint(
            time_ns=self.get_clock().now().nanoseconds,
            robot_configuration=q_sensor,
            robot_velocity=v_sensor,
            robot_acceleration=np.zeros_like(v_sensor),
        )
        reference_trajectory = self.traj_buffer.horizon
        if len(self.ocp.input_transforms) > 0:
            # We assume that TF buffer has been given enough time between setup_mpc and setup_mpc_initial_guess
            # for the transforms to be available.
            self.update_transforms()
        self.ocp.set_reference_weighted_trajectory(reference_trajectory)

        reference_trajectory_points = [el.point for el in reference_trajectory]
        x0, x_init, u_init = ws_ref.generate(initial_state, reference_trajectory_points)
        self.ocp.solve(x0, x_init, u_init, use_iteration_limits_and_timeout=False)
        self._ws_shift.update_previous_solution(self.ocp.ocp_results)

    def sensor_callback(self, sensor_msg: Sensor) -> None:
        """Update the sensor_msg attribute of the class."""
        self.sensor_msg = sensor_msg

    def mpc_input_callback(self, msg: MpcInput) -> None:
        """Fill the new point msg in the trajectory buffer."""
        w_traj_point = mpc_msg_to_weighted_traj_point(
            msg, self.get_clock().now().nanoseconds
        )
        self.traj_buffer.append(w_traj_point)
        self.params.ocp.effector_frame_name = msg.ee_inputs[0].frame_id
        self.effector_frame_name = msg.ee_inputs[0].frame_id

    def update_geom_pose_callback(self, pose: Pose, geom_name: str):
        """Updates pose of a geometry with a given name."""
        if self.mpc is None:
            self.get_logger().warn(
                f"Trying to update geom '{geom_name}', "
                "but the MPC was not initialized yet...",
                throttle_duration_sec=5.0,
            )
            return

        self.ocp.update_geometry_placement(geom_name, ros_pose_to_se3(pose))

    def buffer_has_enough_data(self, ratio: float) -> bool:
        """
        Return True if buffer size is more than `ratio` times
        the horizon size and False otherwise.
        """
        return (
            len(self.traj_buffer) * self.ocp_params.dt
            >= ratio * self.ocp_params.total_time
        )

    def send_control_msg(self, ocp_res: OCPResults) -> None:
        """Get OCP control output and publish it."""
        assert self.np_sensor_msg is not None

        # With a planar base (nu = 3 + n_arm), the feedforward and Ricatti gains
        # include base DoFs that the LFC does not handle.
        # Strip the base slice so the LFC receives arm-only commands (joint_nv).
        # For free_flyer and fixed-base robots, base_nv = 0 or the actuation model
        # already produces arm-only controls (ActuationModelFloatingBase), so no
        # slicing is needed.
        base_nv = 3 if self.params.planar_base else 0
        feedforward = ocp_res.feed_forward_terms[0][base_nv:]
        ricatti_gain = ocp_res.ricatti_gains[0][base_nv:, :]

        if self.params.planar_base:
            # MPC gain is (n_arm, ndx_planar), LFC expects (n_arm, ndx_ff).
            # Map planar tangent cols to FF tangent cols, zero-padding vz/wx/wy.
            # Planar ndx: [q_base_diff(nv_base), q_arm(n_arm), v_base(nv_base), v_arm(n_arm)]
            # LFC FF ndx: [q_ff_diff(nv_ff),     q_arm(n_arm), v_ff(nv_ff),     v_arm(n_arm)]
            nv_base = base_nv  # 3 for planar [vx, vy, wz]
            nv_joints = feedforward.shape[0]
            nv_ff = 6  # free-flyer nv, hardcoded in LFC (free_flyer_nv_ = 6)
            nv_lfc = nv_ff + nv_joints

            K_lfc = np.zeros((nv_joints, 2 * nv_lfc))
            # q_diff: planar [vx,vy,wz] → FF [0,1,5] (vz=2,wx=3,wy=4 stay zero)
            K_lfc[:, [0, 1, 5]] = ricatti_gain[:, 0:nv_base]
            K_lfc[:, nv_ff : nv_ff + nv_joints] = ricatti_gain[
                :, nv_base : nv_base + nv_joints
            ]
            # v_diff: same layout, offset by nv_lfc
            K_lfc[:, [nv_lfc + 0, nv_lfc + 1, nv_lfc + 5]] = ricatti_gain[
                :, nv_base + nv_joints : 2 * nv_base + nv_joints
            ]
            K_lfc[:, nv_lfc + nv_ff : nv_lfc + nv_ff + nv_joints] = ricatti_gain[
                :, 2 * nv_base + nv_joints : 2 * nv_base + 2 * nv_joints
            ]
            ricatti_gain = K_lfc

        ctrl_msg = lfc_py_types.Control(
            feedback_gain=ricatti_gain,
            feedforward=feedforward.reshape(len(feedforward), 1),
            initial_state=self.np_sensor_msg,
        )
        self.control_publisher.publish(control_numpy_to_msg(ctrl_msg))

        if self.params.planar_base:
            base_u = ocp_res.feed_forward_terms[0][:3]
            twist = Twist()
            twist.linear.x = base_u[0]
            twist.linear.y = base_u[1]
            twist.angular.z = base_u[2]
            self.base_cmd_vel_publisher.publish(twist)

    def initialization_callback(self) -> None:
        if self.sensor_msg is None:
            self.get_logger().warn(
                "Waiting for sensor messages to arrive...",
                throttle_duration_sec=5.0,
            )
            return

        if not self.ros_robot_ready():
            return

        # Fetch moving_joint_names from LFC asynchronously (non-blocking).
        if self.moving_joint_names is None:
            if self._lfc_param_future is None:
                if not self._lfc_param_client.service_is_ready():
                    self.get_logger().warn(
                        "Waiting for LFC parameter service...",
                        throttle_duration_sec=5.0,
                    )
                    return
                req = GetParameters.Request()
                req.names = ["moving_joint_names"]
                self._lfc_param_future = self._lfc_param_client.call_async(req)
            if not self._lfc_param_future.done():
                return
            result = self._lfc_param_future.result()
            self._lfc_param_future = None
            if result is None:
                self.get_logger().error(
                    "Failed to get moving_joint_names from LFC, retrying..."
                )
                return
            self.moving_joint_names = result.values[0].string_array_value
            self.get_logger().info(
                f"Got moving_joint_names from LFC: {self.moving_joint_names}"
            )

        if self.mpc is None:
            self.create_robot_models(
                free_flyer=self.params.free_flyer,
                planar_base=self.params.planar_base,
                collision_as_capsule=self.params.collision_as_capsule,
                self_collision=self.params.self_collision,
                armature=self.params.ocp.armature,
                collision_pairs=self.params.collision_pairs,
                robot_attachment_frame=self.params.robot_attachment_frame,
            )
            self.setup_mpc()
            # It is necessary to return, even if the reference buffer has enough data.
            # This gives some time to fill TF buffer.
            return

        if self.params.publish_buffer_size:
            self.ocp_buffer_size_pub.publish(Int32(data=len(self.traj_buffer)))

        # Wait for enough data in buffer
        if not self.buffer_has_enough_data(2):
            self.get_logger().warn(
                f"MPC is waiting for the buffer to be filled... Current size {len(self.traj_buffer)}",
                throttle_duration_sec=5.0,
            )
            return

        # Stop the initialization loop.
        self.destroy_timer(self._init_timer)
        self.setup_mpc_initial_guess()

        self.get_logger().info(
            "MPC is initialized and buffer has enough data. Starting to compute controls."
        )
        # Start the MPC loop
        self.create_timer(1.0 / self.params.rate, self.run_callback)

    def run_callback(self, *args) -> None:
        """
        Timer callback that checks we can start solve before doing it,
        then publish messages related to the OCP.
        """
        if self.params.constant_delay:
            if self._ocp_res is not None:
                # Publish the result from previous OCP.
                self.send_control_msg(self._ocp_res)
                control = self._ocp_res.feed_forward_terms[0].copy()
                self._ocp_res = None
            else:
                control = None

        # Check the buffer size.
        # If size is inferior to `1.5 * required size`, send a warning message. Note that the
        # ratio here must be strictly inferior to the ratio that triggers starting.
        # If size is inferior to the required size, send an error message and return.
        if not self.buffer_has_enough_data(1.5):
            if self.buffer_has_enough_data(1):
                self.get_logger().warn(
                    f"MPC is running and the buffer size becomes low. Current size {len(self.traj_buffer)}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.get_logger().error(
                    f"MPC is running but the buffer does not have enough data. Current size {len(self.traj_buffer)}",
                    throttle_duration_sec=1.0,
                )
                self.traj_buffer.append(self.traj_buffer[-1])
        if self.params.publish_debug_data:
            # Do not use ROS time here because we want to measure the real computation time
            start_compute_time = time.perf_counter()
        self.np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)

        # Update the input transforms required by the OCP, if any.
        self.update_transforms()

        q_arm = self.np_sensor_msg.joint_state.position
        v_arm = self.np_sensor_msg.joint_state.velocity
        if self.params.planar_base:
            q_sensor, v_sensor = sensor_ff_to_planar_state(
                self.sensor_msg, q_arm, v_arm
            )
        else:
            q_sensor, v_sensor = q_arm, v_arm

        x0_traj_point = TrajectoryPoint(
            time_ns=self.get_clock().now().nanoseconds,
            robot_configuration=q_sensor,
            robot_velocity=v_sensor,
            robot_acceleration=np.zeros_like(v_sensor),
        )
        if self.params.constant_delay and control is not None:
            # Compensate for delay by estimating the future state.
            x0_traj_point = self.mpc.integrate(x0_traj_point, control)
            # Update np_sensor_msg so that the published message contains the correct initial state
            self.np_sensor_msg.joint_state.position = x0_traj_point.robot_configuration
            self.np_sensor_msg.joint_state.velocity = x0_traj_point.robot_velocity
        ocp_res = self.mpc.run(
            initial_state=x0_traj_point,
            # Use x0_traj_point time so that this corresponds to time in the future
            # when using delay compensation.
            current_time_ns=x0_traj_point.time_ns,
        )

        if self.params.publish_buffer_size:
            self.ocp_buffer_size_pub.publish(Int32(data=len(self.traj_buffer)))

        if ocp_res is None:
            return

        if self.params.constant_delay:
            self._ocp_res = ocp_res
        else:
            self.send_control_msg(ocp_res)

        if self.params.publish_debug_data:
            compute_time = time.perf_counter() - start_compute_time
            self.ocp_solve_time_pub.publish(Duration(seconds=compute_time).to_msg())
            self.ocp_x0_pub.publish(self.sensor_msg)
            mpc_debug_msg = mpc_debug_data_to_msg(self.mpc.mpc_debug_data)
            self.mpc_debug_pub.publish(mpc_debug_msg)


def main(args=None) -> None:
    """Creates the Agimus controller ROS node object and spins it."""
    rclpy.init(args=args)
    agimus_controller_node = AgimusController()
    try:
        rclpy.spin(agimus_controller_node)
    except KeyboardInterrupt:
        pass
    agimus_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
