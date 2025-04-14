import numpy as np
import numpy.typing as npt
import pinocchio as pin

from agimus_controller.trajectories.quintic_trajectory import QuinticTrajectory
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class SinusWaveConfigurationSpace:
    def __init__(
        self,
        w,
        scale_duration: np.float64,
        amp: np.float64,
        ee_frame_name,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
    ):
        self.quint_traj = QuinticTrajectory(scale_duration=scale_duration, amp=amp)
        self.w = w
        self.ee_frame_name = ee_frame_name
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose
        self.ee_frame_id = None
        self.pin_model = None
        self.pin_data = None
        self.q0 = None
        self.q = None
        self.dq = None
        self.ddq = None

    def set_init_configuration(self, q0: npt.NDArray[np.float64]) -> None:
        """Set q0 of the robot."""
        self.q0 = q0
        self.q = self.q0.copy()
        self.dq = np.zeros_like(self.q)
        self.ddq = np.zeros_like(self.q)

    def set_pin_model(self, pin_model: pin.Model) -> None:
        """Set pinocchio model of the robot and frame id."""
        self.pin_model = pin_model
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        amp, damp, ddamp = self.quint_traj.get_value_at_t(t)
        w = self.w
        sin_wt = np.sin(w * t)
        cos_wt = np.cos(w * t)
        for i in [2, 4]:
            self.q[i] = self.q0[i] + amp * sin_wt
            self.dq[i] = damp * sin_wt + amp * w * cos_wt
            self.ddq[i] = ddamp * sin_wt + 2 * damp * w * cos_wt - amp * w * w * sin_wt
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
