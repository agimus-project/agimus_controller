from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pinocchio as pin

from agimus_controller.trajectory import WeightedTrajectoryPoint


class TrajectoryBase(ABC):
    """Base class for the Trajectory generator class .

    This class defines the interface for the Trajectory generator."""

    def __init__(self, ee_frame_name) -> None:
        self.ee_frame_name = ee_frame_name
        self.trajectory_is_done = False
        self.ee_frame_id = None
        self.pin_model = None
        self.pin_data = None
        self.q0 = None
        self.q = None
        self.dq = None
        self.ddq = None
        self.is_initialized = False

    def initialize(self, pin_model: pin.Model, q0: npt.NDArray[np.float64]) -> None:
        """Initialize the trajectory generator."""
        self.pin_model = pin_model
        self.pin_data = self.pin_model.createData()
        assert self.pin_model.existFrame(self.ee_frame_name), "Frame does not exist."
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
        self.q0 = q0
        self.q = self.q0.copy()
        self.dq = np.zeros(self.pin_model.nv)
        self.ddq = np.zeros(self.pin_model.nv)
        self.is_initialized = True

    def get_end_effector_pose_from_q_as_se3(self, q):
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)
        return self.pin_data.oMf[self.ee_frame_id].copy()

    def get_end_effector_pose_from_q(
        self, q: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.get_end_effector_pose_from_q_as_se3(q)

    @abstractmethod
    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        """Return Weighted Trajectory point of the trajectory at time t."""
        pass
