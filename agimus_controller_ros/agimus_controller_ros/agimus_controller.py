#!/usr/bin/env python3
import numpy as np


import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterValue

from std_msgs.msg import String
from agimus_msgs.msg import MpcInput, MpcDebug
import builtin_interfaces

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
)


from agimus_controller.trajectory import TrajectoryBuffer, TrajectoryPoint
from agimus_controller_ros.agimus_controller_parameters import agimus_controller_params



def get_param_from_node(node: Node, node_name: str, param_name: str) -> ParameterValue:
    """Returns parameter from a node"""
    service_name = f"/{node_name}/get_parameters"
    param_client = node.create_client(GetParameters, service_name)
    while not param_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(f"Service {service_name} not available, waiting again...")
    request = GetParameters.Request()
    request.names = [param_name]

    future = param_client.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        return future.result().values[0]
    else:
        raise ValueError(f"Failed to get parameter {param_name} from node {node_name}")


class RobotMixin:
    def init_ros_robot_creation(self) -> None:
        self.q0 = None
        self.robot_description_msg = None
        self.environment_msg = None
        self.robot_srdf_description_msg = None

        # Get moving joint names from LFC
        self.moving_joint_names = get_param_from_node(self,
            "linear_feedback_controller", "moving_joint_names"
        ).string_array_value

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
        return self.q0 is not None

    def create_robot_models(self, **robot_model_parameters_kwargs) -> None:
        robot_params = RobotModelParameters(
            robot_urdf=self.robot_description_msg.data,
            env_urdf=self.environment_msg.data,
            srdf=self.robot_srdf_description_msg.data,
            moving_joint_names=self.moving_joint_names,
            **robot_model_parameters_kwargs
        )
        self.robot_models = RobotModels(robot_params)
        self.rmodel = self.robot_models._robot_model

        self.get_logger().info("Robot Models initialized")

class AgimusController(Node, RobotMixin):
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
        self.ocp_params = OCPParamsBaseCroco(
            dt=self.params.ocp.dt,
            dt_factor_n_seq=self.params.ocp.dt_factor_n_seq,
            horizon_size=self.params.ocp.horizon_size,
            solver_iters=self.params.ocp.max_iter,
            callbacks=self.params.ocp.activate_callback,
            qp_iters=self.params.ocp.max_qp_iter,
            use_debug_data=self.params.publish_debug_data,
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
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.control_publisher = self.create_publisher(
            Control,
            "control",
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

    def setup_mpc(self):
        """Creates mpc, ocp, warmstart"""
        yaml_file = self.params.ocp.definition_yaml_file
        if yaml_file == "":
            yaml_file = OCPCrocoGeneric.get_default_yaml_file("ocp_goal_reaching.yaml")
        ocp = OCPCrocoGeneric(self.robot_models, self.ocp_params, yaml_file)

        ws_shift = WarmStartShiftPreviousSolution()
        ws_shift.setup(self.robot_models, self.ocp_params)

        # Use WarmStartReference for initialization
        ws_ref = WarmStartReference()
        ws_ref.setup(self.robot_models._robot_model)

        np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)
        initial_state = TrajectoryPoint(
            time_ns=self.get_clock().now().nanoseconds,
            robot_configuration=np_sensor_msg.joint_state.position,
            robot_velocity=np_sensor_msg.joint_state.velocity,
            robot_acceleration=np.zeros_like(np_sensor_msg.joint_state.velocity),
        )
        reference_trajectory = self.traj_buffer.horizon
        ocp.set_reference_weighted_trajectory(reference_trajectory)

        reference_trajectory_points = [el.point for el in reference_trajectory]
        x0, x_init, u_init = ws_ref.generate(initial_state, reference_trajectory_points)
        ocp.solve(x0, x_init, u_init, use_iteration_limits_and_timeout=False)
        ws_shift.update_previous_solution(ocp.ocp_results)

        self.mpc = MPC()
        self.mpc.setup(ocp=ocp, warm_start=ws_shift, buffer=self.traj_buffer)

    def sensor_callback(self, sensor_msg: Sensor) -> None:
        """Update the sensor_msg attribute of the class."""
        self.sensor_msg = sensor_msg

    def mpc_input_callback(self, msg: MpcInput) -> None:
        """Fill the new point msg in the trajectory buffer."""
        w_traj_point = mpc_msg_to_weighted_traj_point(
            msg, self.get_clock().now().nanoseconds
        )
        self.traj_buffer.append(w_traj_point)
        self.params.ocp.effector_frame_name = msg.ee_frame_name
        self.effector_frame_name = msg.ee_frame_name

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
        ctrl_msg = lfc_py_types.Control(
            feedback_gain=ocp_res.ricatti_gains[0],
            feedforward=ocp_res.feed_forward_terms[0].reshape(self.rmodel.nv, 1),
            initial_state=self.np_sensor_msg,
        )
        self.control_publisher.publish(control_numpy_to_msg(ctrl_msg))

    def initialization_callback(self) -> None:
        if self.sensor_msg is None:
            self.get_logger().warn(
                "Waiting for sensor messages to arrive...",
                throttle_duration_sec=5.0,
            )
            return

        if not self.ros_robot_ready():
            self.get_logger().warn(
                "Waiting for robot descriptions...",
                throttle_duration_sec=5.0,
            )
            return

        # Wait for enough data in buffer
        if not self.buffer_has_enough_data(2):
            self.get_logger().warn(
                f"MPC is waiting for the buffer to be filled... Current size {len(self.traj_buffer)}",
                throttle_duration_sec=5.0,
            )
            return

        # Stop the initialization loop.
        self.destroy_timer(self._init_timer)
        self.create_robot_models(
            free_flyer=self.params.free_flyer,
            collision_as_capsule=self.params.collision_as_capsule,
            self_collision=self.params.self_collision,
            armature=self.params.ocp.armature,
            collision_pairs=self.params.collision_pairs,
        )
        self.setup_mpc()

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
                return
        if self.params.publish_debug_data:
            start_compute_time = self.get_clock().now()
        self.np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)

        x0_traj_point = TrajectoryPoint(
            time_ns=self.get_clock().now().nanoseconds,
            robot_configuration=self.np_sensor_msg.joint_state.position,
            robot_velocity=self.np_sensor_msg.joint_state.velocity,
            robot_acceleration=np.zeros_like(self.np_sensor_msg.joint_state.velocity),
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
        if ocp_res is None:
            return

        if self.params.constant_delay:
            self._ocp_res = ocp_res
        else:
            self.send_control_msg(ocp_res)
        if self.params.publish_debug_data:
            compute_time = self.get_clock().now() - start_compute_time
            self.ocp_solve_time_pub.publish(compute_time.to_msg())
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
