import numpy as np
import numpy.typing as npt

import pinocchio as pin

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    """
    A class for generating warmstart values for trajectory optimization problem.

    This class uses a reference trajectory and the robot model to compute the initial state,
    state vectors, and control inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        # The robot's Pinocchio model, used for forward dynamics computations.
        self._rmodel: pin.Model | None = None
        # Data structure associated with the Pinocchio model.
        self._rdata: pin.Data | None = None
        # Size of the state vector
        self._nx: int = 0

    def setup(self, rmodel: pin.Model) -> None:
        self._rmodel = rmodel
        self._rdata = self._rmodel.createData()
        self._nx = self._rmodel.nq + self._rmodel.nv

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
        """
        # Ensure the robot model (_rmodel) is initialized before proceeding
        assert self._rmodel is not None, (
            "Robot model is missing in warmstart. please use warmstart.setup(rmodel)"
        )

        x0 = np.concatenate(
            [initial_state.robot_configuration, initial_state.robot_velocity]
        )
        assert x0.shape[0] == (self._nx), (
            f"Expected x0 shape {(self._nx)},from provided reference got {x0.shape}"
        )

        x_init = [
            np.hstack([point.robot_configuration, point.robot_velocity])
            for point in [initial_state] + reference_trajectory
        ]

        assert np.array(x_init).shape == (
            len(reference_trajectory) + 1,
            self._nx,
        ), (
            f"Expected x_init shape {(len(reference_trajectory) + 1, self._nx)}, "
            f"from provided reference got {np.array(x_init).shape}"
        )
        u_init = [
            pin.rnea(
                self._rmodel,
                self._rdata,
                point.robot_configuration,
                point.robot_velocity,
                point.robot_acceleration,
            )
            # reduce the size of control ref by one to fit Croco way of doing things
            for point in [initial_state] + reference_trajectory[:-1]
        ]
        assert np.array(u_init).shape == (
            len(reference_trajectory),
            self._rmodel.nv,
        ), (
            f"Expected u_init shape {(len(reference_trajectory), self._rmodel.nv)}, "
            f"from provided reference got {np.array(u_init).shape}"
        )

        return x0, x_init, u_init
