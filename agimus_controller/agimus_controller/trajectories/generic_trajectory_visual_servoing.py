import numpy as np
import pinocchio as pin

from agimus_controller.trajectories.generic_trajectory import GenericTrajectory
from agimus_controller.trajectory import TrajectoryPointWeights, WeightedTrajectoryPoint


class GenericTrajectoryVisualServoing(GenericTrajectory):
    """Trajectory class that awaits for trajectory inputs by the user."""

    def __init__(
        self,
        ee_frame_name,
        traj_params,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
    ):
        super().__init__(ee_frame_name, w_q, w_qdot, w_qddot, w_robot_effort, w_pose)
        self.w_pose_constant = w_pose
        # if distance to goal is below this threshold, start visual servoing
        self.start_visual_servoing_distance_threshold = (
            traj_params.start_visual_servoing_distance_threshold
        )
        self.max_weight = traj_params.increasing_weights.max_weight
        self.percent = traj_params.increasing_weights.percent
        self.time_reach_percent = traj_params.increasing_weights.time_reach_percent
        self.simulate_happypose = traj_params.simulate_happypose
        # current visual servoing pose
        self.visual_servoing_pose = None
        # boolean to decides to activate visual servoing
        self.activate_visual_servoing = False
        # Wether current trajectory requires visual servoing or not
        self.use_visual_servoing = False
        self.visual_servoing_time = 0.0
        self.init_in_world_M_object = None

    def get_increasing_weight(self):
        return self.max_weight

    def set_visual_servoing_pose(self, visual_servoing_pose):
        # TODO Only change translation part for now, rotation has to be tested
        self.visual_servoing_pose[:3] = visual_servoing_pose[:3]

    def update_activation_of_visual_servoing(self):
        if self.use_visual_servoing:
            current_ee_pose = next(
                iter(self.trajectory[self.traj_idx].end_effector_poses.values())
            )
            goal_ee_pose = next(iter(self.trajectory[-1].end_effector_poses.values()))
            distance_to_goal = np.linalg.norm(current_ee_pose[:3] - goal_ee_pose[:3])
            self.activate_visual_servoing = (
                distance_to_goal < self.start_visual_servoing_distance_threshold
            )
        else:
            self.activate_visual_servoing = False
        if not self.activate_visual_servoing:
            self.visual_servoing_time = 0.0

    def add_trajectory(
        self, trajectory, use_visual_servoing, init_in_world_M_object=None
    ):
        if use_visual_servoing:
            if init_in_world_M_object is None:
                raise ValueError("Init pose detection not set.")
            self.init_in_world_M_object = pin.XYZQUATToSE3(init_in_world_M_object)
        super().add_trajectory(trajectory)
        self.use_visual_servoing = use_visual_servoing

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        self.update_activation_of_visual_servoing()
        traj_point = self.trajectory[self.traj_idx]
        key = next(iter(traj_point.end_effector_poses))
        if self.init_in_world_M_object is not None:
            in_world_M_ee = pin.XYZQUATToSE3(traj_point.end_effector_poses[key])
            in_object_M_ee = self.init_in_world_M_object.inverse() * in_world_M_ee
            traj_point.end_effector_poses[key] = pin.SE3ToXYZQUAT(in_object_M_ee)
        if self.activate_visual_servoing:
            # if self.visual_servoing_pose is None:
            #    raise ValueError("Visual Servoing pose is not set.")

            self.w_pose = [self.get_increasing_weight()] * 6
            self.visual_servoing_time += 0.01
        else:
            self.w_pose = self.w_pose_constant

        self.trajectory_is_done = self.traj_idx == len(self.trajectory) - 1
        self.traj_idx = min(self.traj_idx + 1, len(self.trajectory) - 1)
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: self.w_pose},
        )
        return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
