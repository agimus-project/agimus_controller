#!/usr/bin/env python3
import numpy as np
import numpy.typing as npt
import time
import pinocchio as pin

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from agimus_msgs.msg import MpcInput
from builtin_interfaces.msg import Duration

import linear_feedback_controller_msgs_py.lfc_py_types as lfc_py_types
from linear_feedback_controller_msgs_py.numpy_conversions import (
    sensor_msg_to_numpy,
    control_numpy_to_msg,
)
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller_ros.sim_utils import (
    convert_float_to_ros_duration_msg,
    mpc_msg_to_weighted_traj_point,
)
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP

from agimus_controller.trajectory import TrajectoryBuffer
from agimus_controller_ros.agimus_controller_parameters import (
    agimus_controller_ros_params,
)


class AgimusController(Node):
    """Agimus controller's ROS 2 node class."""

    def __init__(self, node_name: str = "agimus_controller_node") -> None:
        """Get ROS parameters, initialize trajectory buffer and ros attributes."""
        super().__init__(node_name)
        self._param_listener = agimus_controller_ros_params.ParamListener(self)
        self.params = self._param_listener.get_params()
        self.params.ocp.armature = np.array(self.params.ocp.armature)
        self.traj_buffer = TrajectoryBuffer()
        self.last_point = None
        self.first_run_done = False

        self.initialize_ros_attributes()
        self.get_logger().info("Init done")

    def initialize_ros_attributes(self) -> None:
        """Initialize ROS related attributes such as Publishers, Subscribers and Timers"""
        self.sensor_msg = None
        self.control_msg = None
        self.state_subscriber = self.create_subscription(
            Sensor,
            "sensor",
            self.sensor_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
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
        self.subscriber_mpc_input = self.create_subscription(
            MpcInput,
            "mpc_input",
            self.mpc_input_callback,
            qos_profile=10,
        )
        self.control_publisher = self.create_publisher(
            Control,
            "control",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.ocp_solve_time_pub = self.create_publisher(Duration, "ocp_solve_time", 10)
        self.ocp_x0_pub = self.create_publisher(Sensor, "ocp_x0", 10)
        self.create_timer(self.params.ocp.dt, self.run_callback)

    def sensor_callback(self, sensor_msg: Sensor) -> None:
        """Update the sensor_msg attribute of the class."""
        self.sensor_msg = sensor_msg

    def mpc_input_callback(self, msg: MpcInput) -> None:
        """Fill the new point msg in the trajectory buffer."""
        w_traj_point = mpc_msg_to_weighted_traj_point(msg)
        self.traj_buffer.append(w_traj_point)
        self.params.ocp.effector_frame_name = msg.ee_frame_name
        self.effector_frame_name = msg.ee_frame_name

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        # Initialize models from URDF String
        self.rmodel = pin.buildModelFromXML(msg.data)
        self.cmodel = pin.buildGeomFromUrdfString(
            self.rmodel, msg.data, geom_type=pin.COLLISION
        )
        self.rdata = self.rmodel.createData()
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.initialize_weighted_traj_attributes()

        # Build reduced models
        locked_joint_names = [
            name
            for name in self.rmodel.names
            if name not in self.params.moving_joint_names and name != "universe"
        ]
        locked_joint_ids = [self.rmodel.getJointId(name) for name in locked_joint_names]
        self.rmodel, geometric_models_reduced = pin.buildReducedModel(
            self.rmodel,
            list_of_geom_models=[self.cmodel],
            list_of_joints_to_lock=locked_joint_ids,
            reference_configuration=np.zeros(self.rmodel.nq),
        )
        self.cmodel = geometric_models_reduced[0]
        self.get_logger().info("Robot Models initialized")

    def initialize_weighted_traj_attributes(self) -> None:
        """Initialize attributes to setup OCP from the weighted trajectory buffer."""
        self.x_plan = np.zeros([self.params.ocp.horizon_size, self.nx])
        self.a_plan = np.zeros([self.params.ocp.horizon_size, self.nv])
        self.weight_q = np.zeros([self.params.ocp.horizon_size, self.nq])
        self.weight_qdot = np.zeros([self.params.ocp.horizon_size, self.nv])
        self.weight_pose = np.zeros([self.params.ocp.horizon_size, 6])

    def set_weighted_traj_attributes(self) -> None:
        """Set attributes to setup OCP from the weighted trajectory buffer."""
        for idx_point in range(self.params.ocp.horizon_size):
            traj_point = self.traj_buffer.popleft()
            self.x_plan[idx_point, :] = np.concatenate(
                [traj_point.point.robot_configuration, traj_point.point.robot_velocity]
            )
            self.a_plan[idx_point, :] = traj_point.point.robot_acceleration
            self.weight_q[idx_point, :] = traj_point.weights.w_robot_configuration
            self.weight_qdot[idx_point, :] = traj_point.weights.w_robot_velocity
            self.weight_pose[idx_point, :] = traj_point.weights.w_end_effector_poses

    def buffer_has_twice_horizon_points(self) -> bool:
        """
        Return true if buffer size has more than two times
        the horizon size and False otherwise.
        """
        return len(self.traj_buffer) >= 2 * self.params.ocp.horizon_size

    def first_solve(self, x0: npt.NDArray[np.float64]) -> None:
        """Initialize OCP, then do the first solve."""
        # Initialize OCP
        self.set_weighted_traj_attributes()
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, self.params.ocp)
        self.ocp.set_planning_variables(self.x_plan, self.a_plan)
        self.ocp.set_weights_variables(
            self.weight_q, self.weight_qdot, self.weight_pose
        )

        # First solve
        self.mpc = MPC(self.ocp, self.x_plan, self.a_plan, self.rmodel, self.cmodel)
        us_init = self.mpc.ocp.u_plan[: self.params.ocp.horizon_size - 1]
        self.mpc.mpc_first_step(self.x_plan, us_init, x0)
        self.next_node_idx = self.params.ocp.horizon_size
        self.get_logger().info("First solve done ")

    def solve(self, x0: npt.NDArray[np.float64]) -> None:
        """get new trajectory point, reset and solve the OCP problem."""
        if len(self.traj_buffer) > 0:
            self.last_point = self.traj_buffer.popleft()
        self.mpc.mpc_step(x0, self.last_point, self.effector_frame_name)

    def send_control_msg(self, sensor_msg: Sensor) -> None:
        """Get OCP control output and publish it."""
        _, tau, K_ricatti = self.mpc.get_mpc_output()
        ctrl_msg = lfc_py_types.Control(
            feedback_gain=K_ricatti,
            feedforward=tau.reshape(self.rmodel.nv, 1),
            initial_state=sensor_msg,
        )
        self.control_publisher.publish(control_numpy_to_msg(ctrl_msg))

    def get_x0_from_sensor_msg(self, sensor_msg: Sensor) -> npt.NDArray[np.float64]:
        """Create x0 array from robot sensor message."""
        return np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

    def run_callback(self, *args) -> None:
        """
        Timer callback that checks we can start solve before doing it,
        then publish messages related to the OCP.
        """
        if self.sensor_msg is None:
            self.get_logger().warn(
                "Waiting for sensor messages to arrive...",
                throttle_duration_sec=5.0,
            )
            return
        start_compute_time = time.time()
        np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)
        if not self.first_run_done:
            if not self.buffer_has_twice_horizon_points():
                self.get_logger().warn(
                    "Waiting for buffer to be filled...",
                    throttle_duration_sec=5.0,
                )
                return
            self.first_solve(self.get_x0_from_sensor_msg(np_sensor_msg))
            self.first_run_done = True
        else:
            self.solve(self.get_x0_from_sensor_msg(np_sensor_msg))
        self.send_control_msg(np_sensor_msg)
        compute_time = time.time() - start_compute_time
        self.ocp_solve_time_pub.publish(convert_float_to_ros_duration_msg(compute_time))
        self.ocp_x0_pub.publish(self.sensor_msg)


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
