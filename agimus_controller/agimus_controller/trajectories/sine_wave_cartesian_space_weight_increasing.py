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
        """Initialize parameters needed for the sine wave in configuration space trajectory."""
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
        self.target_1_pose = None
        self.target_2_pose = None
        self.cycle_duration = 4.0

    def get_targets_time(self, t):
        """get cycle's targets time."""
        cycle_start_time = (
            int(t / self.cycle_duration) * self.cycle_duration
        )  # Date of cycle start in ms

        # Compute the absolute time of the shooting interval, modulo the cycle time,
        # so that 0<=time_a0<self.cycle_duration and 0<time_a1<=self.cycle_duration
        time_target_1 = t - cycle_start_time
        # absolute data of the time of the start of the shooting interval
        if time_target_1 > self.cycle_duration:
            time_target_1 -= self.cycle_duration

        # Compute the absolute time of the shooting interval for the second task, modulo the cycle time,
        # so that 0<=time_b0<self.cycle_duration and 0<time_b1<=self.cycle_duration and [time_a0,time_a1] is in antiphase with [time_b0,time_b1].

        if time_target_1 < self.cycle_duration / 2.0:
            time_target_2 = time_target_1 + self.cycle_duration / 2.0

        else:
            time_target_2 = time_target_1 - self.cycle_duration / 2.0
        return (time_target_1, time_target_2)

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        quint, dquint, _ = self.quint_traj.get_value_at_t(t)
        sin_wt = np.sin(self.w * t)
        cos_wt = np.cos(self.w * t)
        time_target_1, time_target_2 = self.get_targets_time(t)
        ee_des_pos = self.ee_init_pos.copy()
        ee_des_pos.translation += self.amp * quint * sin_wt

        ee_des_vel = np.zeros(6)
        ee_des_vel[:3] = self.amp * (dquint * sin_wt + quint * self.w * cos_wt)
        q, dq = self.inverse_kinematics(ee_des_pos, ee_des_vel)
        if time_target_1 < time_target_2:
            ee_des_pos.translation = self.ee_init_pos.translation + self.amp * quint
        else:
            ee_des_pos.translation = self.ee_init_pos.translation - self.amp * quint

        u = pin.rnea(self.pin_model, self.pin_data, q, dq, self.ddq)
        w_pose = np.array(
            [self.w_increasing.get_weight_at_t(max(time_target_1, time_target_2))] * 6
        )
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
