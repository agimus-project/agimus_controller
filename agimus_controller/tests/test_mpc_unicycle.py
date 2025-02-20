import unittest

from agimus_controller.mpc import MPC

import crocoddyl
import numpy as np
from agimus_controller.ocp_base import OCPBase, OCPDebugData, OCPResults
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.trajectory import (
    TrajectoryPoint,
    WeightedTrajectoryPoint,
    TrajectoryPointWeights,
    TrajectoryBuffer,
)
from agimus_controller.warm_start_base import WarmStartBase


class OCPUnicycle(OCPBase):
    def __init__(self, horizon: int = 20, dt: float = 0.1):
        super().__init__()

        self._models = [crocoddyl.ActionModelUnicycle() for _ in range(horizon)]
        for i, model in enumerate(self._models[:-1]):
            model.costWeights = np.array([1.0, 1.0])
            model.dt = dt

        self._models[-1].costWeights = np.array([100.0, 100.0])
        self._models[-1].dt = dt

        self._problem = crocoddyl.ShootingProblem(
            np.array([0.0, 0.0, 0.0]), self._models[:-1], self._models[-1]
        )
        self._ddp = crocoddyl.SolverDDP(self._problem)

        self._debug_data = OCPDebugData(None, None, None, None)
        self._results = None
        self._ref_traj_callback = None

    def set_reference_horizon_callback(self, ref_traj_callback):
        self._ref_traj_callback = ref_traj_callback

    def set_reference_weighted_trajectory(self, ref_w_traj):
        if self._ref_traj_callback is not None:
            self._ref_traj_callback(ref_w_traj)

    @property
    def n_controls(self):
        # n_controls is the number of running models
        return self._problem.T

    @property
    def dt(self):
        return self._models[0].dt

    def solve(self, x0, x_init, u_init):
        self._problem.x0 = x0
        self._debug_data.problem_solved = self._ddp.solve(x_init, u_init)

        self._results = OCPResults(
            list(self._ddp.xs), list(self._ddp.k), list(self._ddp.us)
        )
        assert len(self._ddp.xs) == len(self._ddp.us) + 1

        self._debug_data.result = [
            TrajectoryPoint(
                time_ns=i * self.dt * 1e9,
                robot_configuration=xs,
                robot_velocity=us,
            )
            for i, (xs, us) in enumerate(zip(self._ddp.xs, self._ddp.us))
        ]
        self._debug_data.result.append(
            TrajectoryPoint(
                time_ns=len(self._ddp.us) * self.dt * 1e9,
                robot_configuration=self._ddp.xs[-1],
                robot_velocity=np.zeros_like(self._ddp.us[-1]),
            )
        )

    @property
    def ocp_results(self):
        return self._results

    @property
    def debug_data(self):
        return self._debug_data


class WarmStartUnicycle(WarmStartBase):
    def __init__(self):
        super().__init__()

    def generate(self, initial_state, reference_trajectory):
        x0 = initial_state.robot_configuration
        if self._previous_solution is None or len(self._previous_solution.states) == 0:
            # No previous solution. Use the reference trajectory.
            x_init = [wpoint.robot_configuration for wpoint in reference_trajectory]
            us = [np.zeros(2) for _ in reference_trajectory[1:]]
        else:
            # Use the previous solution as warm start. For the last point, use the reference trajectory.
            x_init = list(self._previous_solution.states[1:]) + [
                reference_trajectory[-1].robot_configuration
            ]
            us = self._previous_solution.feed_forward_terms
        return x0, x_init, us

    def setup(self):
        pass


def unicycle_integrate(q, dq, dt):
    x, y, theta = q
    v, w = dq
    return q + np.array([v * np.cos(theta), v * np.sin(theta), w]) * dt


def unicycle_plot(x):
    import matplotlib.pyplot as plt

    sc, delta = 0.1, 0.1
    a, b, th = x[0], x[1], x[2]
    c, s = np.cos(th), np.sin(th)
    refs = [
        plt.arrow(
            a - sc / 2 * c - delta * s,
            b - sc / 2 * s + delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
        plt.arrow(
            a - sc / 2 * c + delta * s,
            b - sc / 2 * s - delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
    ]
    return refs


def unicycle_plot_solution(xs, figIndex=1, show=True):
    import matplotlib.pyplot as plt

    plt.figure(figIndex, figsize=(6.4, 6.4))
    for x in xs:
        unicycle_plot(x)
    plt.axis([-2, 2.0, -2.0, 2.0])
    if show:
        plt.show()


class TestMPCUnicycle(unittest.TestCase):
    def test_hoziron_and_dt(self):
        ocp = OCPUnicycle(100, 0.01)

        assert ocp.n_controls == 99
        assert ocp.dt == 0.01

    def test_reference_trajectory(self):
        ocp = OCPUnicycle(100, 0.01)

        warm_start = WarmStartUnicycle()
        mpc = MPC()
        dt_factor_n_seq = DTFactorsNSeq(factors=[1], n_steps=[ocp.n_controls])
        mpc.setup(
            ocp,
            warm_start,
            TrajectoryBuffer(dt_factor_n_seq),
        )

        dt_ns = int(ocp.dt * 1e9)

        N_iter = 500

        for k in range(N_iter + ocp.n_controls + 1):
            mpc.append_trajectory_point(
                WeightedTrajectoryPoint(
                    point=TrajectoryPoint(
                        time_ns=(k - 1) * dt_ns,
                        robot_configuration=np.array([k, 0.0, 0.0]),
                        robot_velocity=np.array([0.0, 0.0, 0.0]),
                    ),
                    weights=TrajectoryPointWeights(
                        w_robot_configuration=np.array([1.0, 1.0, 1.0]),
                        w_robot_velocity=np.array([1.0, 1.0, 1.0]),
                    ),
                )
            )

        # variable iteration in this callback is the loop index in the for loop below.
        # The first joint of the robot configuration is set to the iteration number.
        # Note that for this to work, the time of the trajectory point and the time passed to mpc.run must be coherent.
        def ref_traj_callback(ref_traj):
            assert ref_traj[0].point.robot_configuration[0] == iteration, (
                f"{ref_traj[0].point.robot_configuration[0]} != {iteration}"
            )

        ocp.set_reference_horizon_callback(ref_traj_callback)

        state = TrajectoryPoint(
            robot_configuration=np.array([0.0, 1.0, 1.57]),
            robot_velocity=np.array([0.0, 0.0]),
        )
        time_ns = 0
        for iteration in range(N_iter):
            mpc.run(state, time_ns)
            time_ns += dt_ns

    def test_solve(self):
        ocp = OCPUnicycle(100, 0.01)

        warm_start = WarmStartUnicycle()
        mpc = MPC()
        dt_factor_n_seq = DTFactorsNSeq(factors=[1], n_steps=[ocp.n_controls])
        mpc.setup(
            ocp,
            warm_start,
            TrajectoryBuffer(dt_factor_n_seq),
        )

        dt_ns = int(ocp.dt * 1e9)

        N_iter = 500
        traj = []
        for k in range(N_iter + ocp.n_controls + 1):
            mpc.append_trajectory_point(
                WeightedTrajectoryPoint(
                    point=TrajectoryPoint(
                        time_ns=k * dt_ns,
                        robot_configuration=np.array([0.0, 0.0, 0.0]),
                        robot_velocity=np.array([0.0, 0.0, 0.0]),
                    ),
                    weights=TrajectoryPointWeights(
                        w_robot_configuration=np.array([1.0, 1.0, 1.0]),
                        w_robot_velocity=np.array([1.0, 1.0, 1.0]),
                    ),
                )
            )

        # At the moment, there is no reference trajectory.
        state = TrajectoryPoint(
            robot_configuration=np.array([0.0, 1.0, 1.57]),
            robot_velocity=np.array([0.0, 0.0]),
        )
        time_ns = 0
        for _ in range(N_iter):
            traj.append(state.robot_configuration)
            res = mpc.run(state, time_ns)
            np.testing.assert_allclose(res.states[0], state.robot_configuration)

            control = res.feed_forward_terms[0]
            next_q = unicycle_integrate(state.robot_configuration, control, ocp.dt)
            np.testing.assert_allclose(res.states[1], next_q)

            # TODO more in-depth tests for `res`

            state.robot_configuration = next_q
            state.robot_velocity = control
            time_ns += dt_ns

        traj.append(state.robot_configuration)

    @unittest.skip(
        "This test only plots the trajectory and is here to help developers that writes unit tests."
    )
    def plot_trajectory(self):
        ocp = OCPUnicycle(100, 0.01)

        warm_start = WarmStartUnicycle()
        mpc = MPC()
        mpc.setup(ocp, warm_start)

        dt_ns = int(ocp.dt * 1e9)

        N_iter = 500
        traj = []
        for k in range(N_iter + ocp.n_controls + 1):
            mpc.append_trajectory_point(
                WeightedTrajectoryPoint(
                    point=TrajectoryPoint(
                        time_ns=k * dt_ns,
                        robot_configuration=np.array([0.0, 0.0, 0.0]),
                        robot_velocity=np.array([0.0, 0.0, 0.0]),
                    ),
                    weight=TrajectoryPointWeights(
                        w_robot_configuration=np.array([1.0, 1.0, 1.0]),
                        w_robot_velocity=np.array([1.0, 1.0, 1.0]),
                    ),
                )
            )

        # At the moment, there is no reference trajectory.
        state = TrajectoryPoint(
            robot_configuration=np.array([0.0, 1.0, 1.57]),
            robot_velocity=np.array([0.0, 0.0]),
        )
        time_ns = 0
        for _ in range(N_iter):
            traj.append(state.robot_configuration)
            print(int(time_ns * 1e-6), state.robot_configuration, state.robot_velocity)
            res = mpc.run(state, time_ns)

            # Integrate the control
            control = res.feed_forward_terms[0]
            next_q = unicycle_integrate(state.robot_configuration, control, ocp.dt)
            np.testing.assert_allclose(res.states[1], next_q)

            state.robot_configuration = next_q
            state.robot_velocity = control
            time_ns += dt_ns

        traj.append(state.robot_configuration)
        print(state.robot_configuration, state.robot_velocity)
        unicycle_plot_solution(traj)


if __name__ == "__main__":
    unittest.main()
