from agimus_msgs.msg import MpcInput
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import pinocchio as pin
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np
from sensor_msgs.msg import JointState


class SimpleTrajectoryPublisher(Node):
    """This is a simple trajectory publisher for a Panda robot."""

    def __init__(self):
        super().__init__("simple_trajectory_publisher")

        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        self.ee_frame_name = "fer_link8"
        self.robot_description_msg = None

        self.q0 = None
        self.q = None
        self.dq = None
        self.ddq = None
        self.t = 0.0
        self.dt = 0.01
        self.croco_nq = 7
        self.amp = 0.2
        self.scale_amp = 0.0
        self.scale_duration = 0.2
        self.w = 0.5 * np.pi

        # Obtained by checking "QoS profile" values in out of:
        # ros2 topic info -v /robot_description
        # ros2 topic info -v /joint_states
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
            JointState,
            "joint_states",
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
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.timer = self.create_timer(
            0.01, self.publish_mpc_input
        )  # Publish at 100 Hz
        self.get_logger().info("Simple trajectory publisher node started.")

    def joint_states_callback(self, joint_states_msg: JointState) -> None:
        """Set joint state reference."""
        self.get_logger().warn("Received the joint states.")
        jpos = np.array(joint_states_msg.position)
        # TODO fix this, temp hac to work from sim
        if np.linalg.norm(jpos) > 1e-2:
            self.q0 = jpos
            self.destroy_subscription(self.state_subscriber)
            self.get_logger().warn(f"Received q0 = {[round(el, 2) for el in self.q0]}.")

    def robot_description_callback(self, msg: String) -> None:
        """Create the models of the robot from the urdf string."""
        self.get_logger().warn("Received robot description.")
        self.robot_description_msg = msg
        self.destroy_subscription(self.subscriber_robot_description_)

    def load_models(self):
        """Callback to get robot description and store to object"""
        self.pin_model = pin.buildModelFromXML(self.robot_description_msg.data)
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        self.q = self.q0.copy()
        self.dq = np.zeros_like(self.q)
        self.ddq = np.zeros_like(self.q)
        self.get_logger().warn(f"Model loaded, pin_model.nq = {self.pin_model.nq}")

    def quintic_trajectory(self, t):
        """Computes a quintic polynomial trajectory from 0 to 1 over duration.

        Args:
            t (float): Current time in range [0, init_duration].
            init_duration (float): Total duration of the trajectory.

        Returns:
            float: Position value at time t.
        """
        if t <= 0:
            return 0.0
        elif t >= self.scale_duration:
            return 1.0

        # Normalize time
        s = t / self.scale_duration
        return 10 * s**3 - 15 * s**4 + 6 * s**5  # Quintic polynomial

    def publish_mpc_input(self):
        """
        Main function to create a dummy mpc input
        Modifies each joint in sin manner with 0.2 rad amplitude
        """

        if (
            self.robot_description_msg is not None
            and self.q0 is not None
            and self.pin_model is not None
        ):
            self.load_models()
        else:
            return

        self.scale_amp = self.quintic_trajectory(self.t)
        amp = self.scale_amp * self.amp
        # Currently not changing the last two joints - fingers
        # for i in range(self.pin_model.nq - 2):
        for i in [2, 3]:
            self.q[i] = self.q0[i] + amp * np.sin(self.w * self.t)
            self.dq[i] = amp * self.w * np.cos(self.w * self.t)
            self.ddq[i] = -amp * self.w * self.w * np.sin(self.w * self.t)

        # Extract the end-effector position and orientation
        pin.forwardKinematics(self.pin_model, self.pin_data, self.q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)

        ee_pose = self.pin_data.oMf[self.ee_frame_id]
        xyz_quatxyzw = pin.SE3ToXYZQUAT(ee_pose)

        u = pin.rnea(self.pin_model, self.pin_data, self.q, self.dq, self.ddq)

        # Create the message
        msg = MpcInput()
        msg.w_q = [1.0] * self.croco_nq
        msg.w_qdot = [1e-2] * self.croco_nq
        msg.w_qddot = [1e-6] * self.croco_nq
        msg.w_robot_effort = [1e-4] * self.croco_nq
        msg.w_pose = [1e-2] * 6

        msg.q = list(self.q[: self.croco_nq])
        msg.qdot = list(self.dq[: self.croco_nq])
        msg.qddot = list(self.ddq[: self.croco_nq])

        msg.robot_effort = list(u[: self.croco_nq])

        pose = Pose()
        pose.position.x = xyz_quatxyzw[0]
        pose.position.y = xyz_quatxyzw[1]
        pose.position.z = xyz_quatxyzw[2]
        pose.orientation.x = xyz_quatxyzw[3]
        pose.orientation.y = xyz_quatxyzw[4]
        pose.orientation.z = xyz_quatxyzw[5]
        pose.orientation.w = xyz_quatxyzw[6]
        msg.pose = pose

        msg.ee_frame_name = self.ee_frame_name

        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published MPC Input: {msg}')
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
