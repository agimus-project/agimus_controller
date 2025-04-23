import rclpy
import rclpy.duration
import sys
import argparse
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import rclpy.time
from builtin_interfaces.msg import Duration as DurationMsg
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose
from agimus_msgs.msg import MpcDebug

from agimus_controller_ros.agimus_controller import (
    RobotModelsMixin,
    get_param_from_node,
)
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_msg_to_numpy
import pinocchio
import eigenpy


def pinocchio_se3_to_geometry_msg_pose(M: pinocchio.SE3, pose: Pose) -> Pose:
    "Store in `pose` message the transform `M`"
    pose.position.x = M.translation[0]
    pose.position.y = M.translation[1]
    pose.position.z = M.translation[2]

    q = eigenpy.Quaternion(M.rotation)
    pose.orientation.x = q.x
    pose.orientation.y = q.y
    pose.orientation.z = q.z
    pose.orientation.w = q.w
    return pose


class MPCDebuggerNode(Node, RobotModelsMixin):
    """ROS node class to assist users of the AgimusController ROS node.

    Features:
    - publishes the current MPC prediction as markers that can be viewed in RViz.
    """

    def __init__(
        self,
        frame_name: str,
        parent_frame_name: str,
        marker_namespace: str,
        marker_size: float,
    ):
        """
        Args:
        - frame_name: name of the frame in the pinocchio Model
        - parent_frame_name: string passed to field `marker.header.frame_id` in the published messages.
        - marker_namespace: set the `marker.ns` field.
        - marker_size: set the `marker.scale` field.
        """
        super().__init__("mpc_debugger_node")
        self._frame_name = frame_name
        self._parent_frame_name = parent_frame_name
        self._marker_size = marker_size
        self._marker_ns = marker_namespace

        self.init_ros_robot_creation()
        mpc_node_name = "agimus_controller_node"
        self._robot_has_free_flyer = get_param_from_node(
            self, mpc_node_name, "free_flyer"
        ).bool_value
        self._init_timer = self.create_timer(0.1, self.initialization_callback)

    def initialization_callback(self):
        if not self.ros_robot_ready():
            self.get_logger().warn(
                "Waiting for robot descriptions...",
                throttle_duration_sec=5.0,
            )
            return

        self.destroy_timer(self._init_timer)

        self.get_logger().info("create robot...")
        self.create_robot_models(free_flyer=self._robot_has_free_flyer)
        frame_name_ok = self.rmodel.existFrame(self._frame_name)
        assert frame_name_ok, f"Frame {self._frame_name} could not be found."
        self.rdata = self.rmodel.createData()
        self._fid = self.rmodel.getFrameId(self._frame_name)

        self._marker_array = MarkerArray()
        self._mpc_debug_sub = self.create_subscription(
            MpcDebug,
            "mpc_debug",
            self.mpc_debug_to_prediction_markers,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self._mpc_markers_pub = self.create_publisher(
            MarkerArray,
            "mpc_states_prediction_markers",
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.get_logger().info("init done")

    def _init_marker_array(self, msg: MpcDebug):
        states = matrix_msg_to_numpy(msg.states_predictions)
        n = states.shape[0]
        for i, state in enumerate(states):
            marker = Marker()
            marker.header.frame_id = self._parent_frame_name
            # The MPC debug message does not have a stamp so
            # it is not possible to correctly set
            # marker.header.stamp
            marker.ns = self._marker_ns
            marker.id = i

            marker.type = Marker.SPHERE

            marker.action = Marker.ADD

            marker.scale.x = self._marker_size
            marker.scale.y = self._marker_size
            marker.scale.z = self._marker_size

            # From red to green, with a increasing transparency.
            marker.color.r = (n - 1 - i) / (n - 1)
            marker.color.g = (i + 1) / (n - 1)
            marker.color.b = 0.0
            marker.color.a = 0.2 + 0.8 * (n - 1 - i) / (n - 1)

            marker.lifetime = DurationMsg(sec=1)
            self._marker_array.markers.append(marker)

    def mpc_debug_to_prediction_markers(self, msg: MpcDebug):
        if len(self._marker_array.markers) == 0:
            self._init_marker_array(msg)

        states = matrix_msg_to_numpy(msg.states_predictions)
        nq = self.rmodel.nq
        for state, marker in zip(states, self._marker_array.markers):
            pinocchio.forwardKinematics(self.rmodel, self.rdata, state[:nq])
            M = pinocchio.updateFramePlacement(self.rmodel, self.rdata, self._fid)
            pinocchio_se3_to_geometry_msg_pose(M, marker.pose)

        self._mpc_markers_pub.publish(self._marker_array)


def main(args=None):
    # Initialize rclpy first to handle ROS 2 arguments
    rclpy.init(args=args)

    # Filter out ROS 2-specific arguments before passing to argparse
    filtered_args = rclpy.utilities.remove_ros_args(args)

    # Use argparse to parse the remaining arguments
    parser = argparse.ArgumentParser(
        "mpc_debugger_node",
        description="This node transforms the MPC debug data into a marker array that can be visualized in RViz.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="name of the frame in the pinocchio Model",
    )
    parser.add_argument(
        "--parent-frame",
        type=str,
        default="world",
        help="string passed to field `marker.header.frame_id` in the published messages.",
    )
    parser.add_argument(
        "--marker-size", type=float, default=0.01, help="set the `marker.scale` field."
    )
    parser.add_argument(
        "--marker-ns",
        type=str,
        default="states_predictions",
        help="set the `marker.ns` field.",
    )

    arguments = parser.parse_args(filtered_args[1:])  # Skip the script name

    node = MPCDebuggerNode(
        frame_name=arguments.frame,
        parent_frame_name=arguments.parent_frame,
        marker_namespace=arguments.marker_ns,
        marker_size=arguments.marker_size,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
