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
        self.ee_frame_id = None
        self.pin_model = None
        self.pin_data = None
        self.q0 = None
        self.q = None
        self.dq = None
        self.ddq = None
        pass

    def set_pin_model(self, pin_model: pin.Model) -> None:
        """Set pinocchio model of the robot and frame id."""
        self.pin_model = pin_model
        self.pin_data = self.pin_model.createData()
        self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)

    def set_init_configuration(self, q0: npt.NDArray[np.float64]) -> None:
        """Set q0 of the robot."""
        self.q0 = q0
        self.q = self.q0.copy()
        self.dq = np.zeros_like(self.q)
        self.ddq = np.zeros_like(self.q)

    @abstractmethod
    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        """Return Weighted Trajectory point of the trajectory at time t."""
        pass
