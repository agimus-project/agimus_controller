import numpy as np
import numpy.typing as npt
import pinocchio as pin

from agimus_controller.trajectories.quintic_trajectory import QuinticTrajectory
from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class SinusWaveConfigurationSpace(TrajectoryBase):
    """ "Define the trajectory of a sine-wave in configuration Space."""

    def __init__(
        self,
        sine_wave_params: SinWaveParams,
        ee_frame_name: str,
        w_q: npt.NDArray[np.float64],
        w_qdot: npt.NDArray[np.float64],
        w_qddot: npt.NDArray[np.float64],
        w_robot_effort: npt.NDArray[np.float64],
        w_pose: npt.NDArray[np.float64],
    ):
        """Initialize parameters needed for the sine wave in configuration space trajectory."""
        super().__init__(ee_frame_name)
        self.quint_traj = QuinticTrajectory(
            scale_duration=sine_wave_params.scale_duration
        )
        self.amp = sine_wave_params.amplitude
        self.w = 2.0 * np.pi / sine_wave_params.period  # pulsation
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        quint, dquint, ddquint = self.quint_traj.get_value_at_t(t)
        w = self.w
        sin_wt = np.sin(w * t)
        cos_wt = np.cos(w * t)
        self.q = self.q0 + self.amp * quint * sin_wt
        self.dq = self.amp * (dquint * sin_wt + quint * w * cos_wt)
        self.ddq = self.amp * (
            ddquint * sin_wt + 2 * dquint * w * cos_wt - quint * w * w * sin_wt
        )
        pin.forwardKinematics(self.pin_model, self.pin_data, self.q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)

        ee_pose = pin.SE3ToXYZQUAT(self.pin_data.oMf[self.ee_frame_id])

        u = pin.rnea(self.pin_model, self.pin_data, self.q, self.dq, self.ddq)
        traj_point = TrajectoryPoint(
            time_ns=t,
            robot_configuration=self.q,
            robot_velocity=self.dq,
            robot_acceleration=self.ddq,
            robot_effort=u,
            end_effector_poses={self.ee_frame_name: ee_pose},
        )
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: self.w_pose},
        )
        return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
