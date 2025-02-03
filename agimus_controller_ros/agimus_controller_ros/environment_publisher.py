import xacro

from std_msgs.msg import String
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class EnvrionmentPublisher(Node):
    def __init__(self):
        super().__init__("environment_publisher")
        self.publisher_ = self.create_publisher(
            String,
            "/environment_description",
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.declare_parameter("environment_path", "")
        self.environment_path = (
            self.get_parameter("environment_path").get_parameter_value().string_value
        )
        self.publish_msg()
        # self.timer = self.create_timer(0.1, self.publish_msg)  # Publish at 100 Hz

    # self.destroy_node()

    def publish_msg(self):
        urdf = xacro.process_file(self.environment_path).toxml()
        msg = String()
        msg.data = urdf
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EnvrionmentPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
