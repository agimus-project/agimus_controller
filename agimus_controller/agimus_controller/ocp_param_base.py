from dataclasses import dataclass, field


@dataclass
class OCPParamsBaseCroco:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP.
    dt: float  # Integration step of the OCP.
    solver_iters: int  # Number of solver iterations.
    dt_factor_n_seq: list[
        tuple[
            int,  # Number of dts between two time steps, the "factor".
            int,  # Number of time steps, the "n".
        ]
    ]
    # Number of controls that the OCP uses.
    # It is derived from `dt_factor_n_seq` and not have to be passed.
    _n_controls: int = field(init=False)
    # Number of time steps in the horizon must be equal to:
    #     sum(sn for _, sn in dt_factor_n_seq).
    # This is for sanity check.
    horizon_size: int
    qp_iters: int = 200  # Number of QP iterations (must be a multiple of 25).
    termination_tolerance: float = (
        1e-3  # Termination tolerance (norm of the KKT conditions).
    )
    eps_abs: float = 1e-6  # Absolute tolerance of the solver.
    eps_rel: float = 0.0  # Relative tolerance of the solver.
    callbacks: bool = False  # Flag to enable/disable callbacks.
    use_filter_line_search = False  # Flag to enable/disable the filter line searchs.

    def __post_init__(self):
        self._n_controls = sum(sn for _, sn in self.dt_factor_n_seq)
        assert self.horizon_size == self._n_controls, (
            f"The horizon size {self.horizon_size} must be equal to the sum of the time steps {self._n_controls}."
        )

    @property
    def timesteps(self) -> tuple[float]:
        return sum(
            ((self.dt * factor,) * number for factor, number in self.dt_factor_n_seq),
            tuple(),
        )

    @property
    def n_controls(self) -> int:
        return self._n_controls
