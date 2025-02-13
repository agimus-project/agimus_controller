from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import TrajectoryPoint


@dataclass
class OCPResults:
    """Output data structure of the MPC."""

    states: list[npt.NDArray[np.float64]]
    ricatti_gains: list[npt.NDArray[np.float64]]
    feed_forward_terms: list[npt.NDArray[np.float64]]


@dataclass
class OCPDebugData:
    # Debug data
    result: OCPResults
    references: list[TrajectoryPoint]
    collision_distance_residuals: list[dict[np.float64]]
    # Solver infos
    kkt_norm: np.float64
    nb_iter: np.int64
    nb_qp_iter: np.int64
    problem_solved: bool = False


@dataclass
class MPCDebugData:
    ocp: OCPDebugData
    # Timers
    duration_iteration_ns: int = 0
    duration_horizon_update_ns: int = 0
    duration_generate_warm_start_ns: int = 0
    duration_ocp_solve_ns: int = 0
