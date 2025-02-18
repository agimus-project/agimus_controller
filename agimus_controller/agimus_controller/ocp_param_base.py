from dataclasses import dataclass, field


@dataclass
class DTFactorsNSeq:
    """
    Time varying steps definition based on integration step of OCP.

    Example:
        Suppose you want to simulate a trajectory of length 0.6s
        but use varying length of integration steps.
        You want the first 2 time steps to have a higher resolution (factor of 2),
        the next 2 time steps to have a standard resolution (factor of 1).
        Then you would define:
        ```python
        factors = [2, 1]
        n_steps = [2, 2]
        ```

        With dt=0.1, this will result in the following series of timesteps:
        [0.1, 0.1, 0.2, 0.2]
        which sum to the total time of 0.6s.
    """

    factors: list[int]  # Number of steps between two time steps, the "factor".
    n_steps: list[int]  # Number of time steps, the "n".


@dataclass
class OCPParamsBaseCroco:
    """Input data structure of the OCP."""

    # Data relevant to solve the OCP
    dt: float  # Integration step of the OCP
    solver_iters: int  # Number of solver iterations
    dt_factor_n_seq: (
        DTFactorsNSeq  # Time varying steps definition based on integration step of OCP
    )
    # Number of controls that the OCP uses.
    # It is derived from `dt_factor_n_seq` and does not have to be passed.
    _n_controls: int = field(init=False)
    # Number of time steps in the horizon must be equal to:
    #     sum(sn for _, sn in dt_factor_n_seq).
    # This is for sanity check.
    horizon_size: int
    # Time steps that are used by OCP
    # It is derived from `dt_factor_n_seq` and does not have to be passed.
    timesteps: tuple[float] = field(init=False)
    # Total time for OCP problem
    # It is derived from `dt_factor_n_seq` and does not have to be passed.
    total_time: float = field(init=False)
    qp_iters: int = 200  # Number of QP iterations (must be a multiple of 25).
    termination_tolerance: float = (
        1e-3  # Termination tolerance (norm of the KKT conditions).
    )
    eps_abs: float = 1e-6  # Absolute tolerance of the solver.
    eps_rel: float = 0.0  # Relative tolerance of the solver.
    callbacks: bool = False  # Flag to enable/disable callbacks.
    use_filter_line_search = False  # Flag to enable/disable the filter line searchs.

    def __post_init__(self):
        self._n_controls = sum(sn for sn in self.dt_factor_n_seq.n_steps)
        self.timesteps = sum(
            (
                (self.dt * factor,) * n_steps
                for factor, n_steps in zip(
                    self.dt_factor_n_seq.factors, self.dt_factor_n_seq.n_steps
                )
            ),
            tuple(),
        )
        self.total_time = sum(self.timesteps)
        assert self.horizon_size == self._n_controls, (
            f"The horizon size {self.horizon_size} must be equal to the sum of the time steps {self._n_controls}."
        )

    @property
    def n_controls(self) -> int:
        return self._n_controls
