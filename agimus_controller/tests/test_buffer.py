from copy import deepcopy
import numpy as np
from random import randint
import unittest

from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class TestTrajectoryBuffer(unittest.TestCase):
    """
    TestOCPParamsCrocoBase unittests parameters settters and getters of OCPParamsBaseCrocoCroco class.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.nv = randint(10, 100)  # Number of dof in the robot velocity
        self.nq = self.nv + 1  # Number of dof in the robot configuration

        self.trajectory_size = 100
        self.horizon_size = 10
        self.dt_factor_n_seq = DTFactorsNSeq(factors=[1], dts=[self.horizon_size - 1])
        self.dt = 0.01
        self.dt_ns = int(1e9 * self.dt)

    def generate_random_weighted_states(self, time_ns):
        """
        Generate random data for the TrajectoryPointWeights.
        """
        return WeightedTrajectoryPoint(
            point=TrajectoryPoint(
                time_ns=time_ns,
                robot_configuration=np.random.random(self.nq),
                robot_velocity=np.random.random(self.nv),
                robot_acceleration=np.random.random(self.nv),
                robot_effort=np.random.random(self.nv),
            ),
            weights=TrajectoryPointWeights(
                w_robot_configuration=np.random.random(self.nv),
                w_robot_velocity=np.random.random(self.nv),
                w_robot_acceleration=np.random.random(self.nv),
                w_robot_effort=np.random.random(self.nv),
            ),
        )

    def test_append_data(self):
        """
        Test adding points to the buffer.
        """
        obj = TrajectoryBuffer(self.dt_factor_n_seq)
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        self.assertEqual(len(obj), times_ns.size)

    def test_clear_past(self):
        """
        Test clearing the past of the buffer.
        """
        obj = TrajectoryBuffer(self.dt_factor_n_seq)
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        obj.clear_past()
        self.assertEqual(len(obj), times_ns.size - 1)
        obj.clear_past()
        self.assertEqual(len(obj), times_ns.size - 2)
        obj.clear_past()
        self.assertEqual(len(obj), times_ns.size - 3)

    def test_compute_horizon_index(self):
        """
        Test computing the time indexes from dt_factor_n_seq.
        """
        dt_factor_n_seq = DTFactorsNSeq(factors=[1, 2, 3, 4, 5], dts=[2, 2, 2, 2, 1])
        obj = TrajectoryBuffer(dt_factor_n_seq)

        indexes_out = obj.compute_horizon_indexes()
        indexes_test = [0, 1, 3, 5, 8, 11, 15, 19, 24, 29]
        np.testing.assert_equal(indexes_out, indexes_test)

    def test_horizon(self):
        """
        Test computing the horizon from the dt_factor_n_seq format.
        """
        obj = TrajectoryBuffer(self.dt_factor_n_seq)
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        horizon = obj.horizon
        np.testing.assert_array_equal(
            deepcopy(horizon),
            obj[: len(horizon)],
        )

    def test_horizon_with_more_complex_dt_factor_n_seq(self):
        """
        Test computing the horizon from complex dt_factor_n_seq.
        """
        dt_factor_n_seq = DTFactorsNSeq(factors=[1, 2, 3, 4, 5], dts=[2, 2, 2, 2, 1])
        horizon_indexes = [0, 1, 3, 5, 8, 11, 15, 19, 24, 29]

        obj = TrajectoryBuffer(dt_factor_n_seq)
        self.assertEqual(horizon_indexes, obj.horizon_indexes)

        # Fill the data in
        times_ns = np.arange(
            0, 30 * self.trajectory_size * self.dt_ns, self.dt_ns, dtype=int
        )
        for time_ns in times_ns:
            obj.append(self.generate_random_weighted_states(time_ns))

        # Get the horizon
        np.testing.assert_array_equal(
            deepcopy(obj.horizon),
            [obj[index] for index in horizon_indexes],
        )


if __name__ == "__main__":
    unittest.main()
