from std_msgs.msg import String
import rclpy
from rclpy.task import Future

from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectories.generic_trajectory_visual_servoing import (
    GenericTrajectoryVisualServoing,
)
from agimus_controller_ros.simple_trajectory_publisher import SimpleTrajectoryPublisher


class TrajectoryPublisherWithVisualServoing(SimpleTrajectoryPublisher):
    """This is a trajectory publisher that handles visual servoing."""

    def __init__(self):
        super().__init__()

    def add_trajectory(
        self,
        trajectory,
        use_visual_servoing,
        object_name,
        init_object_pose,
        activate_visual_servoing_idx,
    ):
        if self.params.trajectory_name == "generic_trajectory_visual_servoing":
            self.object_name = object_name
            self.trajectory.add_trajectory(
                trajectory,
                use_visual_servoing,
                init_in_world_M_object=init_object_pose,
                activate_visual_servoing_idx=activate_visual_servoing_idx,
            )
            self.future_trajectory_done = Future()
        else:
            raise RuntimeError(
                f"the function add_trajectory can't be used with trajectory type {self.params.trajectory_name}"
            )

    def get_trajectory(self, trajectory_name: String) -> TrajectoryBase:
        """Build chosen trajectory."""
        if trajectory_name == "generic_trajectory_visual_servoing":
            return GenericTrajectoryVisualServoing(
                ee_frame_name=self.ee_frame_name,
                traj_params=self.params.generic_trajectory_visual_servoing,
                dt=self.params.dt,
                w_q=self.get_weights(self.params.w_q, self.croco_nq),
                w_qdot=self.get_weights(self.params.w_qdot, self.croco_nq),
                w_qddot=self.get_weights(self.params.w_qddot, self.croco_nq),
                w_robot_effort=self.get_weights(
                    self.params.w_robot_effort, self.croco_nq
                ),
                w_pose=self.get_weights(self.params.w_pose, 6),
                w_increasing=self.w_increasing,
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
