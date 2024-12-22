import numpy as np
import numpy.typing as npt
from typing_extensions import override

import pinocchio as pin

from agimus_controller.warm_start_base import WarmStartBase
from agimus_controller.trajectory import TrajectoryPoint


class WarmStartReference(WarmStartBase):
    """
    A class for generating warmstart values for trajectory optimization problem.

    This class uses a reference trajectory and the robot model to compute the initial state,
    state vectors, and control inputs.

    Attributes:
    _rmodel (pin.Model | None): The robot's Pinocchio model, used for forward dynamics computations.
    _rdata (pin.Data | None): Data structure associated with the Pinocchio model.

    Methods:
        setup(rmodel: pin.Model) -> None:
            Initializes the robot model and its associated data structure for later computations.

        generate(
            initial_state: TrajectoryPoint,
            reference_trajectory: list[TrajectoryPoint],
        ) -> tuple[
            npt.NDArray[np.float64],
            list[npt.NDArray[np.float64]],
            list[npt.NDArray[np.float64]],
        ]:
            Generates the initial state, reference state vectors, and control inputs for warmstart.

            Parameters:
                initial_state (TrajectoryPoint): The starting state of the robot, containing joint configuration
                    and velocity information.
                reference_trajectory (list[TrajectoryPoint]): A sequence of desired trajectory points, each containing
                    joint configuration, velocity, and acceleration.

            Returns:
                tuple: A tuple containing:
                    - x0 (npt.NDArray[np.float64]): The initial state vector `[q, v]` where `q` is the joint configuration
                    and `v` is the joint velocity.
                    - x_init (list[npt.NDArray[np.float64]]): A list of state vectors `[q, v]` constructed from the
                    reference trajectory.
                    - u_init (list[npt.NDArray[np.float64]]): A list of control inputs computed using inverse dynamics
                    (RNEA) based on the reference trajectory.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rmodel: pin.Model | None = None
        self._rdata: pin.Data | None = None

    def setup(self, rmodel: pin.Model) -> None:
        self._rmodel = rmodel
        self._rdata = self._rmodel.createData()

    @override
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
        assert (
            self._rmodel is not None
        ), "Robot model is missing in warmstart. please use warmstart.setup(rmodel)"

        x0 = np.concatenate(
            [initial_state.robot_configuration, initial_state.robot_velocity]
        )
        assert x0.shape[0] == (self._rmodel.nq + self._rmodel.nv), (
            f"Expected x0 shape {(self._rmodel.nq + self._rmodel.nv)},"
            f"from provided reference got {x0.shape}"
        )

        x_init = np.array(
            [
                np.hstack([point.robot_configuration, point.robot_velocity])
                for point in reference_trajectory
            ]
        )
        assert x_init.shape == (
            len(reference_trajectory),
            self._rmodel.nq + self._rmodel.nv,
        ), (
            f"Expected x_init shape {(len(reference_trajectory), self._rmodel.nq + self._rmodel.nv)}, "
            f"from provided reference got {x_init.shape}"
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
        assert np.array(u_init).shape == (len(reference_trajectory), self._rmodel.nv), (
            f"Expected u_init shape {(len(reference_trajectory), self._rmodel.nv)}, "
            f"from provided reference got {u_init.shape}"
        )

        return x0, x_init, u_init
