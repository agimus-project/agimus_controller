import numpy as np
import pinocchio as pin

from agimus_controller.trajectories.weight_increasing import WeightIncreasing
from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
from agimus_controller.trajectory import TrajectoryPointWeights, WeightedTrajectoryPoint

from enum import Enum


class VisualServoingState(Enum):
    """Enum of the possible states for visual servoing."""

    IDLE = 1
    USING_VISUAL_SERVOING = 2
    COMING_BACK_TO_IDLE = 3


class GenericVisualServoingTrajectory(GenericTrajectory):
    """
    Trajectory class that awaits for trajectory inputs by the user, and
    can enable visual servoing.
    """

    def __init__(
        self,
        ee_frame_name,
        traj_params,
        dt,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
        w_increasing: WeightIncreasing,
    ):
        super().__init__(ee_frame_name, w_q, w_qdot, w_qddot, w_robot_effort, w_pose)
        self.w_pose_constant = w_pose
        self.w_increasing = w_increasing
        self.w_increasing_max_rotation = traj_params.w_increasing_max_rotation
        self.visual_servoing_state = VisualServoingState.IDLE
        self.dt = dt
        self.visual_servoing_time = 0.0
        self.init_in_world_M_object = None
        self.robot_frame = self.ee_frame_name + "_vs"

        # if distance to goal is below this threshold, start visual servoing
        self.start_visual_servoing_time_threshold = (
            traj_params.start_visual_servoing_time_threshold
        )

        # current trajectory indexes range that specifies when visual trajectory starts and ends
        self.visual_servoing_idx_range = (0, 0)

    def update_activation_of_visual_servoing(self):
        """Update visual servoing state machine."""
        if (
            self.visual_servoing_idx_range[0]
            <= self.traj_idx
            < self.visual_servoing_idx_range[1]
        ):
            if self.visual_servoing_state != VisualServoingState.USING_VISUAL_SERVOING:
                self.visual_servoing_time = 0.0
            self.visual_servoing_state = VisualServoingState.USING_VISUAL_SERVOING
        elif self.visual_servoing_time > 0.0:
            self.visual_servoing_state = VisualServoingState.COMING_BACK_TO_IDLE
        else:
            self.visual_servoing_state = VisualServoingState.IDLE

    def add_trajectory(
        self, trajectory, visual_servoing_idx_range, init_in_world_M_object=None
    ):
        if (
            init_in_world_M_object is None
            and visual_servoing_idx_range[0] != visual_servoing_idx_range[1]
        ):
            raise ValueError("Init pose detection not set.")
        if init_in_world_M_object is not None:
            self.init_in_world_M_object = pin.XYZQUATToSE3(init_in_world_M_object)
        super().add_trajectory(trajectory)
        self.visual_servoing_idx_range = visual_servoing_idx_range
        self.traj_idx = 0
        self.trajectory = trajectory

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        self.update_activation_of_visual_servoing()
        traj_point = self.trajectory[self.traj_idx]
        key = next(iter(traj_point.end_effector_poses))
        if self.init_in_world_M_object is not None:
            in_world_M_ee = pin.XYZQUATToSE3(traj_point.end_effector_poses[key])
            in_object_M_ee = self.init_in_world_M_object.inverse() * in_world_M_ee
            traj_point.end_effector_poses[key] = pin.SE3ToXYZQUAT(in_object_M_ee)
        if self.visual_servoing_state == VisualServoingState.USING_VISUAL_SERVOING:
            w_increasing = self.w_increasing.get_weight_at_t(self.visual_servoing_time)
            w_rot_increasing = (
                w_increasing
                * self.w_increasing_max_rotation
                / self.w_increasing.max_weight
            )
            self.w_pose = [w_increasing] * 3 + [w_rot_increasing] * 3
            self.visual_servoing_time = min(
                self.visual_servoing_time + 0.01, self.w_increasing.time_reach_percent
            )
        elif self.visual_servoing_state == VisualServoingState.COMING_BACK_TO_IDLE:
            w_increasing = self.w_increasing.get_weight_at_t(self.visual_servoing_time)
            w_rot_increasing = (
                w_increasing
                * self.w_increasing_max_rotation
                / self.w_increasing.max_weight
            )
            self.w_pose = [w_increasing] * 3 + [w_rot_increasing] * 3
            self.visual_servoing_time -= 0.01
        else:
            self.w_pose = self.w_pose_constant
        self.trajectory_is_done = self.traj_idx == len(self.trajectory) - 1
        self.traj_idx = min(self.traj_idx + 1, len(self.trajectory) - 1)
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.robot_frame: self.w_pose},
        )
        return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
