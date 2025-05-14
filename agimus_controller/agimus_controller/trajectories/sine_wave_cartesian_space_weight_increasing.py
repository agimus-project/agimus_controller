from copy import deepcopy
import numpy as np
import pinocchio as pin

from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)
from agimus_controller.trajectories.weight_increasing import (
    WeightIncreasing,
)
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class SinusWaveCartesianSpaceWeightIncreasing(SinusWaveCartesianSpace):
    """
    Define the trajectory of a sine-wave in cartesian space using weights
    increasing over time in a cycle for the end-effector cost.
    """

    def __init__(
        self,
        sine_wave_params: SinWaveParams,
        w_increasing: WeightIncreasing,
        ee_frame_name,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
        mask=(True, True, True, True, True, True),
    ):
        """Initialize parameters needed for the sine wave in cartesian space with increasing weight trajectory."""
        super().__init__(
            sine_wave_params,
            ee_frame_name,
            w_q,
            w_qdot,
            w_qddot,
            w_robot_effort,
            w_pose,
            mask,
        )
        self.w_increasing = w_increasing
        self.cycle_durations = sine_wave_params.period

    def get_targets_time(self, t, cycle_duration):
        """get cycle's targets time."""
        cycle_start_time = int(t / cycle_duration) * cycle_duration
        time_target_1 = t - cycle_start_time
        if time_target_1 > cycle_duration:
            time_target_1 -= cycle_duration
        if time_target_1 < cycle_duration / 2.0:
            time_target_2 = time_target_1 + cycle_duration / 2.0
        else:
            time_target_2 = time_target_1 - cycle_duration / 2.0
        return (time_target_1, time_target_2)

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        quint, dquint, _ = self.quint_traj.get_value_at_t(t)
        sin_wt = np.sin(self.w * t)
        cos_wt = np.cos(self.w * t)
        ee_des_vel = np.zeros(6)
        ee_des_vel[:3] = self.amp * (dquint * sin_wt + quint * self.w * cos_wt)

        # compute sine wave inverse kinematics
        sin_wave_ee_des_pos = self.ee_init_pos.copy()
        sin_wave_ee_des_pos.translation += self.amp * quint * sin_wt
        q, dq = self.inverse_kinematics(sin_wave_ee_des_pos, ee_des_vel)

        # compute end-effector pose as a switch between the two extrema of the sine wave with increasing weight
        ee_des_pos = self.ee_init_pos.copy()
        w_pose = self.w_pose
        for ax_idx in range(3):
            time_target_1, time_target_2 = self.get_targets_time(
                t, self.cycle_durations[ax_idx]
            )
            if time_target_1 < time_target_2:
                ee_des_pos.translation[ax_idx] += self.amp[ax_idx] * quint[ax_idx]
            else:
                ee_des_pos.translation[ax_idx] -= self.amp[ax_idx] * quint[ax_idx]
            w_pose[ax_idx] = self.w_increasing.get_weight_at_t(
                max(time_target_1, time_target_2)
            )

        u = pin.rnea(self.pin_model, self.pin_data, q, dq, self.ddq)
        traj_point = TrajectoryPoint(
            time_ns=t,
            robot_configuration=q,
            robot_velocity=dq,
            robot_acceleration=self.ddq,
            robot_effort=u,
            end_effector_poses={self.ee_frame_name: pin.SE3ToXYZQUAT(ee_des_pos)},
        )
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: w_pose},
        )
        return WeightedTrajectoryPoint(
            point=deepcopy(traj_point), weights=deepcopy(traj_weights)
        )
