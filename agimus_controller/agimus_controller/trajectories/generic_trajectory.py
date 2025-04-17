import pinocchio as pin
import numpy as np
import numpy.typing as npt

from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class GenericTrajectory(TrajectoryBase):
    """Trajectory class that awaits for trajectory inputs by the user."""

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

    def build_trajectory_from_q_dq_ddq_arrays(
        self,
        q_array: list[npt.NDArray[np.float64]],
        dq_array: list[npt.NDArray[np.float64]],
        ddq_array: list[npt.NDArray[np.float64]],
    ) -> list[TrajectoryPoint]:
        """Builds list of Trajectory points based on given trajectory of q,dq and ddq."""
        assert len(q_array) == len(dq_array) and len(q_array) == len(ddq_array)
        length = len(q_array)
        trajectory = []
        for idx in range(length):
            robot_effort = pin.rnea(
                self.pin_model,
                self.pin_data,
                q_array[idx],
                dq_array[idx],
                ddq_array[idx],
            )
            pin.forwardKinematics(self.pin_model, self.pin_data, self.q)
            pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)
            ee_pose = pin.SE3ToXYZQUAT(self.pin_data.oMf[self.ee_frame_id])
            trajectory.append(
                TrajectoryPoint(
                    robot_configuration=q_array[idx],
                    robot_velocity=dq_array[idx],
                    robot_acceleration=ddq_array[idx],
                    robot_effort=robot_effort,
                    end_effector_poses={self.ee_frame_name: ee_pose},
                )
            )
        return trajectory

    def add_trajectory(self, trajectory: list[TrajectoryPoint]) -> None:
        """Initialize the trajectory if it wasn't, otherwise extend the trajectory."""
        self.trajectory_is_done = False
        if self.trajectory is None:
            self.trajectory = list(trajectory)
        else:
            self.trajectory.extend(list(trajectory))

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        traj_point = self.trajectory[self.traj_idx]
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
