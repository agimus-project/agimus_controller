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
        mask=6 * [True],
    ):
        """Initialize parameters needed for the sine wave in configuration space trajectory.

        Args:
            sine_wave_params (SinWaveParams): Sine wave parameters: period, amplitude, scale_duration.
            ee_frame_name (_type_): Name of the end effector frame.
            w_q (_type_): weight for the robot configuration.
            w_qdot (_type_): weight for the robot velocity.
            w_qddot (_type_): weight for the robot acceleration.
            w_robot_effort (_type_): weight for the robot effort.
            w_pose (_type_): weight for the end effector pose.
            mask (_type_, optional): Inverse kinematics DoF mask [x, y, z, roll, pitch, yaw]. Defaults to 6*[True].
        """

        super().__init__(ee_frame_name)
        self.quint_traj = QuinticTrajectory(
            scale_duration=sine_wave_params.scale_duration
        )
        self.amp = np.array(sine_wave_params.amplitude)
        self.w = np.array(sine_wave_params.pulsation)
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose
        self.mask = mask
        self.ik_q = None

    def initialize(self, pin_model: pin.Model, q0: np.ndarray) -> None:
        """Initialize the trajectory generator."""
        super().initialize(pin_model, q0)
        self.ik_q = self.q0.copy()
        self.ee_init_pos = self.get_end_effector_pose_from_q_as_se3(self.q0)

    def inverse_kinematics(
        self,
        ee_des_pos: pin.SE3,
        ee_des_vel: np.ndarray,
        precision=1e-3,
        it_max=10000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the inverse kinematics of the robot to reach the desired end effector pose."""
        i = 0
        success = False
        while True:
            self.ik_ee_pose = self.get_end_effector_pose_from_q_as_se3(self.ik_q)
            dMi = ee_des_pos.actInv(self.ik_ee_pose)
            error = pin.log(dMi).vector[self.mask]
            if norm(error) < precision:
                success = True
                break
            if i > it_max:
                break

            pin.computeJointJacobians(self.pin_model, self.pin_data, self.ik_q)
            Jee = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )[self.mask, :]
            dq = Jee.T @ solve(Jee @ Jee.T, error)
            self.ik_q[:] = pin.integrate(self.pin_model, self.ik_q, dq)
            i += 1

        if not success:
            error_msgs = (
                f"Inverse kinematics 6D failed to converge with error: "
                f"{self.ik_error}. Number of iteration: {i}"
            )
            raise RuntimeError(error_msgs)

        pin.forwardKinematics(self.pin_model, self.pin_data, self.ik_q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)
        pin.computeJointJacobians(self.pin_model, self.pin_data, self.ik_q)
        Jee = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[self.mask, :]
        dq = -Jee.T @ solve(Jee @ Jee.T, ee_des_vel[self.mask])
        return self.ik_q.copy(), dq.copy()

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        quint, dquint, _ = self.quint_traj.get_value_at_t(t)
        sin_wt = np.sin(self.w * t)
        cos_wt = np.cos(self.w * t)

        ee_des_pos = self.ee_init_pos.copy()
        ee_des_vel = np.zeros(6)
        ee_des_pos.translation += self.amp * quint * sin_wt
        ee_des_vel[:3] = self.amp * (dquint * sin_wt + quint * self.w * cos_wt)
        q, dq = self.inverse_kinematics(ee_des_pos, ee_des_vel)
        ddq = np.zeros(self.pin_model.nv)
        u = pin.rnea(self.pin_model, self.pin_data, q, dq, ddq)
        traj_point = TrajectoryPoint(
            time_ns=t,
            robot_configuration=q,
            robot_velocity=dq,
            robot_acceleration=ddq,
            robot_effort=u,
            end_effector_poses={self.ee_frame_name: pin.SE3ToXYZQUAT(ee_des_pos)},
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
