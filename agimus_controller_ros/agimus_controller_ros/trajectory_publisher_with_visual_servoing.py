from typing import Tuple

import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String
import rclpy
from rclpy.task import Future
from rclpy.qos import (
    qos_profile_system_default,
)
from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection2DArray, Detection2D

from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectories.generic_trajectory_visual_servoing import (
    GenericTrajectoryVisualServoing,
)
from agimus_controller_ros.ros_utils import ros_pose_to_array
from agimus_controller_ros.simple_trajectory_publisher import SimpleTrajectoryPublisher


def map_object_id(obj_id, dataset="tless"):
    num_part = obj_id.split("_")[1]
    return f"{dataset}-obj_{int(num_part):06d}"


def get_most_confident_object_pose(
    detection_msg: Detection2DArray, object_name: str
) -> Tuple[str, list[float]]:
    # TODO: change the map if we want to use YCBV
    filtered_detections = [
        (d, d.results[0].hypothesis.score)
        for d in detection_msg.detections
        if d.results[0].hypothesis.class_id == map_object_id(object_name)
    ]
    if len(filtered_detections) == 0:
        return ["", None]
    detection: Detection2D = max(filtered_detections, key=lambda pair: pair[1])[0]
    pose: Pose = detection.results[0].pose.pose
    return [
        detection.header.frame_id,
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    ]


class TrajectoryPublisherWithVisualServoing(SimpleTrajectoryPublisher):
    """This is a trajectory publisher that handles visual servoing."""

    def __init__(self):
        super().__init__()
        if (
            self.params.trajectory_name == "generic_trajectory_visual_servoing"
            and not self.params.generic_trajectory_visual_servoing.simulate_happypose
        ):
            self.object_name = None
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.vision_client = self.create_subscription(
                Detection2DArray,
                "/happypose/detections",
                self.vision_callback,
                qos_profile=qos_profile_system_default,
            )

    def add_trajectory(self, trajectory, use_visual_servoing, object_name):
        if self.params.trajectory_name == "generic_trajectory_visual_servoing":
            self.object_name = object_name
            self.trajectory.add_trajectory(trajectory, use_visual_servoing)
            self.future_trajectory_done = Future()
        else:
            raise RuntimeError(
                f"the function add_trajectory can't be used with trajectory type {self.params.trajectory_name}"
            )

    def set_visual_servoing_pose(self, visual_servoing_pose):
        if self.params.trajectory_name == "generic_trajectory_visual_servoing":
            self.trajectory.set_visual_servoing_pose(visual_servoing_pose)

    def vision_callback(self, vision_msg: Detection2DArray):
        if vision_msg.detections == []:
            return
        frame, in_camera_pose_object = get_most_confident_object_pose(
            vision_msg, self.object_name
        )
        if in_camera_pose_object[1] is None:
            raise ValueError(f"No {self.object_name} object detected")
        image_timestamp = vision_msg.detections[0].header.stamp
        in_world_M_camera = self.tf_buffer.lookup_transform(
            target_frame="support_link",
            source_frame=frame,
            time=image_timestamp,
        )
        in_world_pose_object = ros_pose_to_array(
            tf2_geometry_msgs.do_transform_pose(
                in_camera_pose_object, in_world_M_camera
            ).pose
        )
        self.trajectory.set_visual_servoing_pose(in_world_pose_object)

    def get_trajectory(self, trajectory_name: String) -> TrajectoryBase:
        """Build chosen trajectory."""
        if trajectory_name == "generic_trajectory_visual_servoing":
            return GenericTrajectoryVisualServoing(
                ee_frame_name=self.ee_frame_name,
                traj_params=self.params.generic_trajectory_visual_servoing,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
            )
        else:
            super().get_trajectory(trajectory_name=trajectory_name)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisherWithVisualServoing()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
