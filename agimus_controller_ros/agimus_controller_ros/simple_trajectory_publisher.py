from typing import List
import numpy as np

from agimus_msgs.msg import MpcInput
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
from rclpy.task import Future
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from linear_feedback_controller_msgs.msg import Sensor

from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller.trajectories.sine_wave_configuration_space import (
    SinusWaveConfigurationSpace,
)
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)
from agimus_controller.trajectories.sine_wave_cartesian_space_weight_increasing import (
    SinusWaveCartesianSpaceWeightIncreasing,
)
from agimus_controller.trajectories.weight_increasing import WeightIncreasing
from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
from agimus_controller_ros.ros_utils import (
    weighted_traj_point_to_mpc_msg,
    get_param_from_node,
    get_params_from_node,
    sensor_ff_to_planar_state,
)
from agimus_controller.trajectories.generic_visual_servoing_trajectory import (
    GenericVisualServoingTrajectory,
)
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


class TrajectoryPublisherBase(Node):
    """Base class for publishing references for agimus controller node.

    It is responsible for:
    - initializing a ROS node.
    - getting the robot initial joint values.
    - creating the ROS publisher.

    When initialization is completed, the function `ready_callback` is called.
    Child classes should reimplement this function to start publishing.
    """

    def __init__(self, name="trajectory_publisher"):
        super().__init__(name)

        self.q0 = None
        self.current_q = None
        self.robot_description_msg = None

        # Obtained by checking "QoS profile" values in out of:
        # ros2 topic info -v /robot_description
        # ros2 topic info -v /sensor
        self.moving_joint_names = get_param_from_node(
            self, "linear_feedback_controller", "moving_joint_names"
        ).string_array_value

        # Fetch base configuration from agimus_controller_node during __init__,
        # i.e. before rclpy.spin starts, so we can use get_params_from_node safely.
        base_params = get_params_from_node(
            self, "agimus_controller_node", ["free_flyer", "planar_base"]
        )
        self._free_flyer = base_params[0].bool_value
        self._planar_base = base_params[1].bool_value

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

        self.timer = self.create_timer(0.1, self._initialize_q0)

    def joint_states_callback(self, msg: Sensor) -> None:
        """Set joint state reference."""
        jpos = np.array(msg.joint_state.position)
        jvel = np.array(msg.joint_state.velocity)
        # TODO fix this, temp hack to work from sim
        joint_idxs = get_joint_idxs(self.moving_joint_names, msg.joint_state)
        arm_q = get_reduced_configuration(jpos, joint_idxs)
        arm_v = get_reduced_configuration(jvel, joint_idxs)
        if self.q0 is None and np.linalg.norm(arm_q) > 1e-2:
            self.q0 = arm_q.copy()
            self.get_logger().info(f"Received q0 = {[round(el, 2) for el in self.q0]}.")
        # With a planar base, the LFC sends base data in base_pose/base_twist fields.
        # Convert to planar representation; before _load_models, fall back to neutral.
        if self._planar_base:
            q_planar, v_planar = sensor_ff_to_planar_state(msg, arm_q, arm_v)
            self.current_q = q_planar
            self.current_dq = v_planar
        else:
            base_q = getattr(self, "_base_q0", np.array([]))
            self.current_q = np.concatenate([base_q, arm_q])
            self.current_dq = np.concatenate([np.zeros(len(base_q)), arm_v])

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        self.get_logger().info("Received robot description.")
        self.robot_description_msg = msg
        self.destroy_subscription(self.subscriber_robot_description_)

    def _load_models(self):
        """Callback to get robot description and store to object"""
        self.robot_models = RobotModels(
            param=RobotModelParameters(
                robot_urdf=self.robot_description_msg.data,
                free_flyer=self._free_flyer,
                planar_base=self._planar_base,
                moving_joint_names=self.moving_joint_names,
            )
        )

        # q0 and current_q from joint_states_callback contain arm joints only.
        # With a floating base, prepend the neutral base configuration so that
        # the full configuration matches robot_model.nq.
        rmodel = self.robot_models.robot_model
        nq_base = rmodel.nq - len(self.moving_joint_names)
        if nq_base > 0:
            import pinocchio as pin

            base_q0 = pin.neutral(rmodel)[:nq_base]
            self.q0 = np.concatenate([base_q0, self.q0])
            self.current_q = np.concatenate([base_q0, self.current_q])
            self._base_q0 = base_q0
        else:
            self._base_q0 = np.array([])

        self.get_logger().info(f"Model loaded, pin_model.nq = {rmodel.nq}")
        self.get_logger().info(f"Model loaded, reduced self.q0 = {self.q0}")

    def _initialize_q0(self):
        if self.robot_description_msg is None or self.q0 is None:
            self.get_logger().info(
                "Wait for robot model and q0", throttle_duration_sec=1.0
            )
            return

        self._load_models()
        self.destroy_timer(self.timer)
        self.ready_callback()

    def ready_callback(self):
        """Child class should reimplement this method, called when
        - the publisher is ready to be used
        - attributes robot_models and q0 are ready to be used
        """
        pass


class SimpleTrajectoryPublisher(TrajectoryPublisherBase):
    """This is a simple trajectory publisher for a Panda robot."""

    def __init__(self):
        super().__init__("simple_trajectory_publisher")

        self.param_listener = trajectory_weights_params.ParamListener(self)

        self.params = self.param_listener.get_params()
        self.ee_frame_name = self.params.ee_frame_name
        self.sine_wave_params = SinWaveParams(
            amplitude=self.params.sine_wave.amplitude,
            period=self.params.sine_wave.period,
            scale_duration=self.params.sine_wave.scale_duration,
        )
        self.w_increasing = WeightIncreasing(
            self.params.w_increasing.max_weight,
            percent=self.params.w_increasing.percent,
            time_reach_percent=self.params.w_increasing.time_reach_percent,
        )
        self._id: int = 0
        self.t = 0.0
        self.dt = get_param_from_node(
            self, "agimus_controller_node", "ocp.dt"
        ).double_value
        self.croco_nq = 7
        self.future_init_done = Future()
        self.future_trajectory_done = Future()

        self.trajectory = self.get_trajectory(self.params.trajectory_name)
        self.get_logger().info("Simple trajectory publisher node started.")

    def ready_callback(self):
        self.timer = self.create_timer(0.01, self.publish_mpc_input)

    def add_trajectory(self, trajectory):
        """Add custom trajectory chunk to publish if trajectory is of type generic_trajectory."""
        if self.params.trajectory_name == "generic_trajectory":
            self.trajectory.add_trajectory(trajectory)
            self.future_trajectory_done = Future()
        else:
            raise RuntimeError(
                f"the function add_trajectory can't be used with trajectory type {self.params.trajectory_name}"
            )

    def add_visual_servoing_trajectory(
        self, trajectory, visual_servoing_idx_range, init_object_pose
    ):
        """
        Add custom trajectory chunk to publish if trajectory is of type
        visual_servoing_generic_trajectory. Visual servoing can be enabled.
        """
        if self.params.trajectory_name == "generic_visual_servoing_trajectory":
            self.trajectory.add_trajectory(
                trajectory,
                visual_servoing_idx_range,
                init_in_world_M_object=init_object_pose,
            )
            self.future_trajectory_done = Future()
        else:
            raise RuntimeError(
                f"the function add_trajectory can't be used with trajectory type {self.params.trajectory_name}"
            )

    def get_sine_wave_parameters(self) -> SinWaveParams:
        """Get sine wave parameters."""
        sine_wave_amplitude = self.params.sine_wave.amplitude
        if len(sine_wave_amplitude) == 1:
            self.params.sine_wave.amplitude = (sine_wave_amplitude[0],) * len(
                self.moving_joint_names
            )
        sine_wave_period = self.params.sine_wave.period
        if len(sine_wave_period) == 1:
            self.params.sine_wave.period = (sine_wave_period[0],) * len(
                self.moving_joint_names
            )
        sine_wave_scale_duration = self.params.sine_wave.scale_duration
        if len(sine_wave_scale_duration) == 1:
            self.params.sine_wave.scale_duration = (sine_wave_scale_duration[0],) * len(
                self.moving_joint_names
            )
        self.sine_wave_parameters = SinWaveParams(
            amplitude=sine_wave_amplitude,
            period=sine_wave_period,
            scale_duration=sine_wave_scale_duration,
        )
        return self.sine_wave_parameters

    def get_trajectory(self, trajectory_name: String) -> TrajectoryBase:
        """Build chosen trajectory."""
        self.sine_wave_parameters = self.get_sine_wave_parameters()
        if trajectory_name == "sine_wave_configuration_space":
            assert len(self.sine_wave_parameters.amplitude) == len(
                self.moving_joint_names
            ), "sine_wave_amplitude and moving_joint_names must have the same length"
            assert len(self.sine_wave_parameters.period) == len(
                self.moving_joint_names
            ), "sine_wave_period and moving_joint_names must have the same length"
            assert len(self.sine_wave_parameters.scale_duration) == len(
                self.moving_joint_names
            ), (
                "sine_wave_scale_duration and moving_joint_names must have the same length"
            )
            return SinusWaveConfigurationSpace(
                sine_wave_params=self.sine_wave_parameters,
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
            assert len(self.sine_wave_parameters.amplitude) == 3, (
                "sine_wave_amplitude length must be 3"
            )
            assert len(self.sine_wave_parameters.period) == 3, (
                "sine_wave_period length must be 3"
            )
            assert len(self.sine_wave_parameters.scale_duration) == 3, (
                "sine_wave_scale_duration length must be 3"
            )
            return SinusWaveCartesianSpace(
                sine_wave_params=self.sine_wave_parameters,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
                mask=self.params.mask,
            )
        elif trajectory_name == "sine_wave_cartesian_space_weight_increasing":
            assert len(self.sine_wave_parameters.amplitude) == 3, (
                "sine_wave_amplitude length must be 3"
            )
            assert len(self.sine_wave_parameters.period) == 3, (
                "sine_wave_period length must be 3"
            )
            assert len(self.sine_wave_parameters.scale_duration) == 3, (
                "sine_wave_scale_duration length must be 3"
            )
            return SinusWaveCartesianSpaceWeightIncreasing(
                sine_wave_params=self.sine_wave_params,
                w_increasing=self.w_increasing,
                ee_frame_name=self.ee_frame_name,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
                mask=self.params.mask,
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
                w_collision_avoidance=self.params.w_collision_avoidance,
            )
        elif trajectory_name == "generic_visual_servoing_trajectory":
            return GenericVisualServoingTrajectory(
                ee_frame_name=self.ee_frame_name,
                traj_params=self.params.generic_trajectory_visual_servoing,
                dt=self.dt,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
                w_increasing=self.w_increasing,
                w_collision_avoidance=self.params.w_collision_avoidance,
            )
        else:
            raise ValueError("Unknown Trajectory.")

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
        if not self.trajectory.is_initialized:
            self.trajectory.initialize(self.robot_models.robot_model, self.q0)
            self.future_init_done.set_result(True)
            return
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
        w_traj_point.point.id = self._id
        msg = weighted_traj_point_to_mpc_msg(w_traj_point)
        self._id += 1

        self.publisher_.publish(msg)
        if self.trajectory.trajectory_is_done:
            self.future_trajectory_done.set_result(True)
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
