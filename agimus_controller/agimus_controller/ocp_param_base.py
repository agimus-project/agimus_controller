from dataclasses import dataclass


@dataclass
class DTFactorsNSeq:
    """Time varying steps definition based on integration step of OCP."""

    factors: list[int]  # Number of dts between two time steps, the "factor".
    dts: list[int]  # Number of time steps, the "n".


@dataclass
class OCPParamsBaseCroco:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP
    dt: float  # Integration step of the OCP
    horizon_size: int  # Number of time steps in the horizon
    solver_iters: int  # Number of solver iterations
    dt_factor_n_seq: DTFactorsNSeq
    qp_iters: int = 200  # Number of QP iterations (must be a multiple of 25).
    termination_tolerance: float = (
        1e-3  # Termination tolerance (norm of the KKT conditions).
    )
    eps_abs: float = 1e-6  # Absolute tolerance of the solver.
    eps_rel: float = 0.0  # Relative tolerance of the solver.
    callbacks: bool = False  # Flag to enable/disable callbacks.
    nb_threads: int = 1  # Number of threads used by OCP solver.
    use_filter_line_search = False  # Flag to enable/disable the filter line searchs.

    def __post_init__(self):
        assert (
            self.horizon_size == sum(sn for sn in self.dt_factor_n_seq.dts) + 1
            and "The horizon size must be equal to the sum of the time steps."
        )

    @property
    def timesteps(self) -> tuple[float]:
        return sum(
            ((self.dt * factor,) * number for factor, number in self.dt_factor_n_seq),
            tuple(),
        )
