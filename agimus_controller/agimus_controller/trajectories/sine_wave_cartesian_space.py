from copy import deepcopy
from numpy.linalg import norm, solve
import numpy as np
import pinocchio as pin

from agimus_controller.trajectories.quintic_trajectory import QuinticTrajectory
from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class SinusWaveCartesianSpace(TrajectoryBase):
    """ "Define the trajectory of a sine-wave in configuration Space."""

    def __init__(
        self,
        sine_wave_params: SinWaveParams,
        ee_frame_name,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
    ):
        """Initialize parameters needed for the sine wave in configuration space trajectory."""
        super().__init__(ee_frame_name)
        self.quint_traj = QuinticTrajectory(
            scale_duration=sine_wave_params.scale_duration
        )
        self.amp = sine_wave_params.amplitude
        self.w = sine_wave_params.pulsation
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose

        # Desired end-effector trajectory parameters.
        self.ee_init_pos = pin.SE3.Identity()
        self.ee_des_pos = pin.SE3.Identity()
        self.ee_des_vel = np.zeros(6)

        # Inverse kinematics parameters.
        self.ik_error = 0.0
        self.ik_precision = 1e-7
        self.it_max = 10000
        self.ik_dt = 1e-1
        self.ik_damp = 1e-6
        self.ik_p_error = np.zeros(3)
        self.ik_success = False
        self.ik_q = None
        self.ik_dq = None
        self.ik_Jp = None
        self.ik_Jdamp = None

    def set_pin_model(self, pin_model: pin.Model) -> None:
        """Set pinocchio model of the robot and frame id."""
        super().set_pin_model(pin_model)
        self.ik_q = pin.neutral(self.pin_model)
        self.ik_dq = np.zeros(self.pin_model.nv)
        self.ik_Jp = np.zeros((6, self.pin_model.nv))
        self.ik_Jdamp = np.eye(6)

    def inverse_kinematics(
        self, ee_des_pos: pin.SE3, ee_des_vel: pin.Motion
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the inverse kinematics of the robot to reach the desired end effector pose."""
        i = 0
        while True:
            pin.forwardKinematics(self.pin_model, self.pin_data, self.ik_q)
            pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)
            dMi = ee_des_pos.actInv(self.pin_data.oMf[self.ee_frame_id])
            self.ik_error = pin.log(dMi).vector
            if norm(self.ik_error) < self.ik_precision:
                self.ik_success = True
                break
            if i > self.it_max:
                self.ik_success = False
                break

            pin.computeJointJacobians(self.pin_model, self.pin_data, self.ik_q)
            self.ik_Jp[:, :] = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            self.ik_dq[:] = -self.ik_Jp.T.dot(
                solve(
                    self.ik_Jp.dot(self.ik_Jp.T) + self.ik_damp * self.ik_Jdamp,
                    self.ik_error,
                )
            )
            self.ik_q[:] = pin.integrate(
                self.pin_model, self.ik_q, self.ik_dq * self.ik_dt
            )
            i += 1

        pin.computeJointJacobians(self.pin_model, self.pin_data, self.ik_q)
        self.ik_Jp[:, :] = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        self.ik_dq[:] = -self.ik_Jp.T.dot(
            solve(
                self.ik_Jp.dot(self.ik_Jp.T) + self.ik_damp * self.ik_Jdamp,
                self.ee_des_vel,
            )
        )
        if not self.ik_success:
            error_msgs = (
                f"Inverse kinematics failed to converge with error: {self.ik_error}."
            )
            print(error_msgs)
            # raise RuntimeError(error_msgs)

        return self.ik_q.copy(), self.ik_dq.copy()

    def set_init_configuration(self, q0: np.ndarray) -> None:
        """Set q0 of the robot."""
        super().set_init_configuration(q0)
        # Compute the initial desired end effector pose.
        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)
        self.ik_q = self.q0.copy()

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        quint, dquint, _ = self.quint_traj.get_value_at_t(t)
        sin_wt = np.sin(self.w * t)
        cos_wt = np.cos(self.w * t)

        self.ee_des_pos.translation = (
            self.ee_init_pos.translation + self.amp * quint * sin_wt
        )
        self.ee_des_vel[:3] = self.amp * (dquint * sin_wt + quint * self.w * cos_wt)
        self.q[:], self.dq[:] = self.inverse_kinematics(
            self.ee_des_pos, self.ee_des_vel
        )

        u = pin.rnea(self.pin_model, self.pin_data, self.q, self.dq, self.ddq)
        traj_point = TrajectoryPoint(
            time_ns=t,
            robot_configuration=self.q,
            robot_velocity=self.dq,
            robot_acceleration=self.ddq,
            robot_effort=u,
            end_effector_poses={self.ee_frame_name: pin.SE3ToXYZQUAT(self.ee_des_pos)},
        )
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: self.w_pose},
        )
        return WeightedTrajectoryPoint(
            point=deepcopy(traj_point), weights=deepcopy(traj_weights)
        )
