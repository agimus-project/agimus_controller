from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import typing as T


@dataclass
class OCPResults:
    """Output data structure of the MPC."""

    states: list[npt.NDArray[np.float64]] = field(default_factory=list)
    ricatti_gains: list[npt.NDArray[np.float64]] = field(default_factory=list)
    feed_forward_terms: list[npt.NDArray[np.float64]] = field(default_factory=list)


@dataclass
class OCPDebugData:
    # Debug data
    result: OCPResults = OCPResults()
    references: list[T.Tuple[str, npt.NDArray[np.float64]]] = field(
        default_factory=list
    )
    residuals: list[T.Tuple[str, T.List[npt.NDArray[np.float64]]]] = field(
        default_factory=list
    )

    # Solver infos
    kkt_norm: np.float64 = 0.0
    nb_iter: np.int64 = 0
    nb_qp_iter: np.int64 = 0
    problem_solved: bool = False


@dataclass
class MPCDebugData:
    ocp: OCPDebugData = OCPDebugData()
    # Timers
    duration_iteration_ns: int = 0
    duration_horizon_update_ns: int = 0
    duration_generate_warm_start_ns: int = 0
    duration_ocp_solve_ns: int = 0
