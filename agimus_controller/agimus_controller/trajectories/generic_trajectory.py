from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class GenericTrajectory(TrajectoryBase):
    """ "Define the trajectory of a sine-wave in configuration Space."""

    def __init__(
        self,
        ee_frame_name,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
    ):
        super().__init__(ee_frame_name)
        self.trajectory = None
        self.traj_idx = 0
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose

    def add_trajectory(self, trajectory):
        if self.trajectory is None:
            self.trajectory = list(trajectory)
        else:
            self.trajectory.extend(list(trajectory))

    def get_traj_point_at_t(self) -> WeightedTrajectoryPoint:
        traj_point = self.trajectory[self.traj_idx]
        self.traj_idx = min(self.traj_idx + 1, len(self.trajectory) - 1)
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: self.w_pose},
        )
        return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
