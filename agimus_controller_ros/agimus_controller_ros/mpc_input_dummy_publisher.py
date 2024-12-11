from agimus_msgs.msg import MpcInput
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from rclpy.node import Node
import rclpy
import pinocchio as pin
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np


class MpcInputDummyPublisher(Node):
    def __init__(self):
        super().__init__("mpc_input_dummy_publisher")

        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        self.ee_frame_name = "fer_joint7"
        # Zero pose from which the motion will start
        self.q0 = np.array([0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78])
        self.q = self.q0.copy()
        self.t = 0.0
        self.dt = 0.01

        # Obtained by checking "QoS profile" values in out of:
        # ros2 topic info -v /robot_description
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.get_logger().info("CREATING subscriber_robot_description_")
        self.subscriber_robot_description_ = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            qos_profile=qos_profile,
        )
        self.publisher_ = self.create_publisher(MpcInput, "mpc_input", 10)
        self.timer = self.create_timer(0.05, self.publish_mpc_input)  # Publish at 20 Hz
        self.get_logger().info("MPC Dummy Input Publisher Node started.")

    def robot_description_callback(self, msg: String):
        """Callback to get robot description and store to object"""
        self.pin_model = pin.buildModelFromXML(msg.data)
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        self.get_logger().warn(f"Model loaded, pin_model.nq = {self.pin_model.nq}")

    def publish_mpc_input(self):
        """
        Main function to create a dummy mpc input
        Modifies each joint in sin manner with 0.2 rad amplitude
        """
        if self.pin_model is None:  # wait for model to be available
            return

        # Currently not changing the last two joints - fingers
        # TODO: change once we have a finger flag
        for i in range(self.pin_model.nq - 2):
            self.q[i] = self.q0[i] + 0.2 * np.sin(2 * np.pi * self.t)

        # Extract the end-effector position and orientation
        pin.forwardKinematics(self.pin_model, self.pin_data, self.q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)

        ee_pose = self.pin_data.oMf[self.ee_frame_id]
        xyz_quatxyzw = pin.SE3ToXYZQUAT(ee_pose)

        # Create the message
        msg = MpcInput()
        msg.q = [float(val) for val in self.q]

        msg.qdot = [0.0] * len(
            self.q
        )  # TODO: only works for robot with only revolute joints
        msg.qddot = [0.0] * len(
            self.q
        )  # TODO: only works for robot with only revolute joints
        msg.q_w = [1.0] * len(self.q)
        msg.qdot_w = [1e-3] * len(self.q)
        msg.qddot_w = [1e-3] * len(self.q)

        pose = Pose()
        pose.position.x = xyz_quatxyzw[0]
        pose.position.y = xyz_quatxyzw[1]
        pose.position.z = xyz_quatxyzw[2]
        pose.orientation.x = xyz_quatxyzw[3]
        pose.orientation.y = xyz_quatxyzw[4]
        pose.orientation.z = xyz_quatxyzw[5]
        pose.orientation.w = xyz_quatxyzw[6]
        msg.pose = pose
        msg.pose_w = [1.0] * 6

        msg.ee_frame_name = self.ee_frame_name

        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published MPC Input: {msg}')
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = MpcInputDummyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
