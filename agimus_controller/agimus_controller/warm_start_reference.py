import numpy as np
import numpy.typing as npt

import pinocchio as pin

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    def __init__(self) -> None:
        super().__init__()
        self._rmodel: pin.Model | None = None
        self._rdata: pin.Data | None = None

    def setup(self, rmodel: pin.Model) -> None:
        self._rmodel = rmodel
        self._rdata = self._rmodel.createData()

    def generate(
        self,
        initial_state: TrajectoryPoint,
        reference_trajectory: list[TrajectoryPoint],
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        list[npt.NDArray[np.float64]],
    ]:
        """
        Generate initial values for a warm-start of the optimization problem.
        The state vector is `[q, v]`, where:
            - `q` is the robot's joint configuration.
            - `v` is the robot's joint velocity.
        - `init_xs`: A list of state vectors `[q, v]` constructed from the reference trajectory.
        - `init_us`: A list of control inputs computed using inverse dynamics (RNEA)
            based on the reference trajectory.

        For more details, see base class.
        """
        # Ensure the robot model (_rmodel) is initialized before proceeding
        assert self._rmodel is not None
        x0 = np.concatenate(
            [initial_state.robot_configuration, initial_state.robot_velocity]
        )

        x_init = np.array(
            [
                np.hstack([point.robot_configuration, point.robot_velocity])
                for point in reference_trajectory
            ]
        )
        u_init = [
            pin.rnea(
                self._rmodel,
                self._rdata,
                point.robot_configuration,
                point.robot_velocity,
                point.robot_acceleration,
            )
            for point in reference_trajectory
        ]
        return x0, x_init, u_init
