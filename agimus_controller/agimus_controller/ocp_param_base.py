from dataclasses import dataclass

import numpy as np


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
