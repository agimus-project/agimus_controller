from typing import List
import numpy as np

from agimus_msgs.msg import MpcInput
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from linear_feedback_controller_msgs.msg import Sensor
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterValue


from agimus_controller.factory.robot_model import (
    RobotModelParameters,
    RobotModels,
)
from agimus_controller.trajectories.sine_wave_configuration_space import (
    SinusWaveConfigurationSpace,
)
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)
from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
from agimus_controller_ros.ros_utils import weighted_traj_point_to_mpc_msg
from agimus_controller_ros.trajectory_weights_parameters import (
    trajectory_weights_params,
)


def get_joint_idxs(
    moving_joint_names: list[str], joint_state: JointState
) -> list[np.int64]:
    idxs = []
    for joint_name in moving_joint_names:
        idxs.append(joint_state.name.index(joint_name))
    return idxs


def get_reduced_configuration(q: list[np.float64], joint_idxs: list[np.int64]):
    reduced_q = np.zeros((len(joint_idxs)))
    for idx, joint_idx in enumerate(joint_idxs):
        reduced_q[idx] = q[joint_idx]
    return reduced_q


class SimpleTrajectoryPublisher(Node):
    """This is a simple trajectory publisher for a Panda robot."""

    def __init__(self):
        super().__init__("simple_trajectory_publisher")

        self.param_listener = trajectory_weights_params.ParamListener(self)
        self.params = self.param_listener.get_params()
        self.ee_frame_name = self.params.ee_frame_name
        self.robot_description_msg = None

        self.q0 = None
        self.current_q = None
        self.t = 0.0
        self.dt = 0.01
        self.croco_nq = 7
        self.trajectory = self.get_trajectory(self.params.trajectory_name)

        # Obtained by checking "QoS profile" values in out of:
        # ros2 topic info -v /robot_description
        # ros2 topic info -v /sensor
        self.moving_joint_names = self.get_param_from_node(
            "linear_feedback_controller", "moving_joint_names"
        ).string_array_value
        self.subscriber_robot_description_ = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.state_subscriber = self.create_subscription(
            Sensor,
            "sensor",
            self.joint_states_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.publisher_ = self.create_publisher(
            MpcInput,
            "mpc_input",
            qos_profile=QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.timer = self.create_timer(
            0.01, self.publish_mpc_input
        )  # Publish at 100 Hz
        self.get_logger().info("Simple trajectory publisher node started.")

    def get_param_from_node(self, node_name: str, param_name: str) -> ParameterValue:
        """Returns parameter from the node"""
        param_client = self.create_client(GetParameters, f"/{node_name}/get_parameters")
        while not param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        request = GetParameters.Request()
        request.names = [param_name]

        future = param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result().values[0]
        else:
            raise ValueError("Failed to load moving joint names from LFC")

    def publish(self, trajectory):
        self.trajectory.add_trajectory(trajectory)

    def joint_states_callback(self, msg: Sensor) -> None:
        """Set joint state reference."""
        jpos = np.array(msg.joint_state.position)
        # TODO fix this, temp hac to work from sim
        joint_idxs = get_joint_idxs(self.moving_joint_names, msg.joint_state)
        if self.q0 is None and np.linalg.norm(jpos) > 1e-2:
            self.q0 = get_reduced_configuration(jpos, joint_idxs)
            self.trajectory.set_init_configuration(q0=self.q0)
            self.get_logger().warn(f"Received q0 = {[round(el, 2) for el in self.q0]}.")
        self.current_q = get_reduced_configuration(jpos, joint_idxs)
        self.current_dq = get_reduced_configuration(
            np.array(msg.joint_state.velocity), joint_idxs
        )

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        self.get_logger().warn("Received robot description.")
        self.robot_description_msg = msg
        self.destroy_subscription(self.subscriber_robot_description_)

    def get_trajectory(self, trajectory_name: String) -> TrajectoryBase:
        """Build chosen trajectory."""
        if trajectory_name == "sine_wave_configuration_space":
            return SinusWaveConfigurationSpace(
                self.params.sine_wave,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        elif trajectory_name == "sine_wave_cartesian_space":
            return SinusWaveCartesianSpace(
                self.params.sine_wave,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        elif trajectory_name == "generic_trajectory":
            return GenericTrajectory(
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        else:
            raise ValueError("Unknown Trajectory.")

    def load_models(self):
        """Callback to get robot description and store to object"""
        self.robot_models = RobotModels(
            param=RobotModelParameters(
                robot_urdf=self.robot_description_msg.data,
                free_flyer=False,
                moving_joint_names=self.moving_joint_names,
            )
        )
        self.trajectory.set_pin_model(self.robot_models.robot_model)

        self.get_logger().warn(
            f"Model loaded, pin_model.nq = {self.trajectory.pin_model.nq}"
        )
        self.get_logger().warn(f"Model loaded, reduced self.q0 = {self.q0}")

    def get_weights(
        self, weights: List[np.float64], size: np.float64
    ) -> List[np.float64]:
        """
        Return weights with right size if user sent only one value, otherwise
        directly returns weights.
        """
        if len(weights) == 1:
            return weights * size
        else:
            return weights

    def publish_mpc_input(self):
        """
        Main function to create a dummy mpc input
        Modifies each joint in sin manner with 0.2 rad amplitude
        """

        if self.robot_description_msg is None or self.q0 is None:
            return

        if self.trajectory.pin_model is None:
            self.load_models()
        if (
            self.params.trajectory_name == "generic_trajectory"
            and self.trajectory.trajectory is None
        ):
            self.get_logger().warn(
                "Waiting for trajectory to be initialized.",
                throttle_duration_sec=5.0,
            )
            return
        w_traj_point = self.trajectory.get_traj_point_at_t(self.t)
        msg = weighted_traj_point_to_mpc_msg(w_traj_point)

        self.publisher_.publish(msg)
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrajectoryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
