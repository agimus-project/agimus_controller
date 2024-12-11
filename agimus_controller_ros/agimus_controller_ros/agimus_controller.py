#!/usr/bin/env python3
import atexit
import numpy as np
import time
import pinocchio as pin
import yaml
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
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

from agimus_controller_ros.sim_utils import convert_float_to_ros_duration_msg
from agimus_controller.utils.pin_utils import get_ee_pose_from_configuration
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller_ros.parameters import AgimusControllerNodeParameters
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class AgimusController(Node):
    def __init__(
        self,
        node_name: str = "agimus_controller_node",
        params: AgimusControllerNodeParameters = None,
    ) -> None:
        super().__init__(node_name)
        if params is None:
            self.set_params()
        else:
            self.params = params
        self.traj_buffer = TrajectoryBuffer()
        self.last_point = None
        self.first_run_done = False

        self.initialize_ros_attributes()
        self.get_logger().info("Init done")

    def initialize_ros_attributes(self):
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
        self.control_publisher = self.create_publisher(Control, "/control", 10)
        self.ocp_solve_time_pub = self.create_publisher(Duration, "ocp_solve_time", 10)
        self.ocp_x0_pub = self.create_publisher(Sensor, "ocp_x0", 10)
        self.create_timer(self.params.ocp.dt, self.first_run_callback)
        self.create_timer(self.params.ocp.dt, self.run_callback)
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def set_params(self):
        """Setting the parameters of the node by reading a yaml file."""
        self.declare_parameter("package_name", "")
        package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        self.declare_parameter("path_to_yaml", "")
        path_to_yaml = (
            self.get_parameter("path_to_yaml").get_parameter_value().string_value
        )
        yaml_path = Path(get_package_share_directory(package_name)) / path_to_yaml
        with open(yaml_path, "r") as file:
            params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
        self.params = AgimusControllerNodeParameters(params)

    def sensor_callback(self, sensor_msg: Sensor) -> None:
        """Update the sensor_msg attribute of the class."""
        self.sensor_msg = sensor_msg

    def mpc_input_callback(self, msg: MpcInput):
        """Fill the new point in the buffer."""
        xyz_quat = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        traj_point = TrajectoryPoint(
            robot_configuration=msg.q,
            robot_velocity=msg.qdot,
            robot_acceleration=msg.qddot,
            end_effector_poses={msg.ee_frame_name: pin.XYZQUATToSE3(xyz_quat)},
        )

        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=msg.q_w,
            w_robot_velocity=msg.qdot_w,
            w_robot_acceleration=msg.qddot_w,
            w_end_effector_poses=msg.pose_w,
        )

        w_traj_point = WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
        self.traj_buffer.append(w_traj_point)

    def robot_description_callback(self, msg: String):
        """Create the models of the robot from the urdf string."""
        self.rmodel = pin.buildModelFromXML(msg.data)
        locked_joint_names = [
            name
            for name in self.rmodel.names
            if name not in self.params.moving_joint_names and name != "universe"
        ]
        locked_joint_ids = [self.rmodel.getJointId(name) for name in locked_joint_names]
        self.cmodel = pin.buildGeomFromUrdfString(
            self.rmodel, msg.data, geom_type=pin.COLLISION
        )
        self.vmodel = pin.buildGeomFromUrdfString(
            self.rmodel, msg.data, geom_type=pin.VISUAL
        )
        self.rmodel, geometric_models_reduced = pin.buildReducedModel(
            self.rmodel,
            list_of_geom_models=[self.cmodel, self.vmodel],
            list_of_joints_to_lock=locked_joint_ids,
            reference_configuration=np.zeros(self.rmodel.nq),
        )
        self.cmodel, self.vmodel = geometric_models_reduced

        self.rdata = self.rmodel.createData()
        self.effector_frame_id = self.rmodel.getFrameId(
            self.params.ocp.effector_frame_name
        )
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.nx = self.nq + self.nv
        self.get_logger().info("Robot Models initialized")

    def buffer_has_twice_horizon_points(self) -> bool:
        return len(self.traj_buffer) >= 2 * self.params.ocp.horizon_size

    def first_solve(self, x0: np.ndarray):
        # retrieve horizon state and acc references
        x_plan = np.zeros([self.params.ocp.horizon_size, self.nx])
        a_plan = np.zeros([self.params.ocp.horizon_size, self.nv])
        for idx_point in range(self.params.ocp.horizon_size):
            traj_point = self.traj_buffer.popleft()
            x_plan[idx_point, :] = np.concatenate(
                [traj_point.point.robot_configuration, traj_point.point.robot_velocity]
            )
            a_plan[idx_point, :] = traj_point.point.robot_acceleration

        # First solve
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, self.params.ocp)
        self.ocp.set_planning_variables(x_plan, a_plan)
        self.mpc = MPC(self.ocp, x_plan, a_plan, self.rmodel, self.cmodel)
        us_init = self.mpc.ocp.u_plan[: self.params.ocp.horizon_size - 1]
        self.mpc.mpc_first_step(x_plan, us_init, x0)
        self.next_node_idx = self.params.ocp.horizon_size
        if self.params.save_predictions_and_refs:
            self.mpc.create_mpc_data()
        _, u, k = self.mpc.get_mpc_output()

        self.get_logger().info("First solve done ")
        return u, k

    def solve(self, x0: np.ndarray):
        self.target_translation_object_to_effector = None
        if len(self.traj_buffer) > 0:
            traj_point = self.traj_buffer.popleft()
            self.last_point = traj_point
            new_x_ref = np.concatenate(
                [traj_point.point.robot_configuration, traj_point.point.robot_velocity]
            )
        else:
            traj_point = self.last_point
            new_x_ref = np.concatenate(
                [traj_point.point.robot_configuration, traj_point.point.robot_velocity]
            )
        new_a_ref = np.array(traj_point.point.robot_acceleration)
        self.in_world_M_effector = get_ee_pose_from_configuration(
            self.rmodel,
            self.rdata,
            self.effector_frame_id,
            new_x_ref[: self.rmodel.nq],
        )
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref, self.in_world_M_effector)
        if self.params.save_predictions_and_refs:
            self.mpc.fill_predictions_and_refs_arrays()
        _, tau, K_ricatti = self.mpc.get_mpc_output()
        return tau, K_ricatti

    def send(self, sensor_msg, tau, K_ricatti):
        ctrl_msg = lfc_py_types.Control(
            feedback_gain=K_ricatti,
            feedforward=tau.reshape(self.rmodel.nv, 1),
            initial_state=sensor_msg,
        )
        self.control_publisher.publish(control_numpy_to_msg(ctrl_msg))

    def exit_handler(self):
        print("saving data")
        np.save("mpc_params.npy", self.params.get_dict())
        if self.params.save_predictions_and_refs:
            np.save("mpc_data.npy", self.mpc.mpc_data)

    def get_x0_from_sensor_msg(self, sensor_msg):
        return np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

    def publish_ocp_solve_time(self, ocp_solve_time: float):
        print("publishing solve time ", ocp_solve_time)
        self.ocp_solve_time_pub.publish(
            convert_float_to_ros_duration_msg(ocp_solve_time)
        )

    def first_run_callback(self):
        if (
            not self.first_run_done
            and self.sensor_msg is not None
            and self.buffer_has_twice_horizon_points()
        ):
            np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)
            start_compute_time = time.time()
            tau, K_ricatti = self.first_solve(
                self.get_x0_from_sensor_msg(np_sensor_msg)
            )
            compute_time = time.time() - start_compute_time
            self.send(np_sensor_msg, tau, K_ricatti)
            self.publish_ocp_solve_time(compute_time)
            self.ocp_x0_pub.publish(self.sensor_msg)
            atexit.register(self.exit_handler)
            self.first_run_done = True

    def run_callback(self, *args):
        if self.first_run_done:
            start_compute_time = time.time()
            np_sensor_msg: lfc_py_types.Sensor = sensor_msg_to_numpy(self.sensor_msg)
            tau, K_ricatti = self.solve(self.get_x0_from_sensor_msg(np_sensor_msg))
            self.send(np_sensor_msg, tau, K_ricatti)
            compute_time = time.time() - start_compute_time
            self.publish_ocp_solve_time(compute_time)
            self.ocp_x0_pub.publish(self.sensor_msg)


def main(args=None):
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
