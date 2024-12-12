from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

from agimus_controller.trajectory import WeightedTrajectoryPoint


@dataclass
class OCPParamsCrocoBase:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP
    dt: np.float64  # Integration step of the OCP
    T: int  # Number of time steps in the horizon
    solver_iters: int  # Number of solver iterations
    qp_iters: int = 200  # Number of QP iterations (must be a multiple of 25).
    termination_tolerance: np.float64 = (
        1e-3  # Termination tolerance (norm of the KKT conditions)
    )
    eps_abs: np.float64 = 1e-6  # Absolute tolerance of the solver
    eps_rel: np.float64 = 0  # Relative tolerance of the solver
    callbacks: bool = False  # Flag to enable/disable callbacks

    # Weights, costs & helpers for the costs & constraints
    WeightedTrajectoryPoints: List[
        WeightedTrajectoryPoint
    ] | None = None  # List of weighted trajectory points
    armature: npt.NDArray[np.float64] | None = None  # Armature of the robot
    ee_name: str | None = None  # Name of the end-effector
