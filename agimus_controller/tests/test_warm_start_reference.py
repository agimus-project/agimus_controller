import unittest

import numpy as np
import example_robot_data as robex
import pinocchio as pin

from agimus_controller.warm_start_reference import WarmStartReference
from agimus_controller.trajectory import TrajectoryPoint


class TestWarmStart(unittest.TestCase):
    def test_initialization(self):
        WarmStartReference()
        # Not sure how relevant the following commented test is:
        # 1. Private attributes should not be used directly.
        # 2. WarmStartReference does not make use of the previous solution.
        # self.assertEqual(ws._previous_solution, None)

    def test_generate(self):
        ws = WarmStartReference()
        num_points = 10
        robot = robex.load("ur5")
        rmodel = robot.model
        rdata = robot.data
        ws.setup(rmodel=rmodel)

        initial_q = np.random.randn(rmodel.nq)
        initial_v = np.random.randn(rmodel.nv)
        initial_a = np.random.randn(rmodel.nv)
        initial_state = TrajectoryPoint(
            robot_configuration=initial_q,
            robot_velocity=initial_v,
            robot_acceleration=initial_a,
        )

        random_qs = np.random.randn(num_points, rmodel.nq)
        random_vs = np.random.randn(num_points, rmodel.nv)
        random_acs = np.random.randn(num_points, rmodel.nv)
        reference_trajectory = [
            TrajectoryPoint(
                robot_configuration=q, robot_velocity=v, robot_acceleration=a
            )
            for q, v, a in zip(random_qs, random_vs, random_acs)
        ]

        # Create the expected stacked array
        expected_x0 = np.concatenate([initial_q, initial_v])
        expected_x_init = np.hstack((random_qs, random_vs))
        expected_x_init = np.vstack([expected_x0, expected_x_init])
        expected_u_init = np.array(
            [pin.rnea(rmodel, rdata, initial_q, initial_v, initial_a)]
            + [
                pin.rnea(rmodel, rdata, q, v, a)
                for q, v, a in zip(random_qs, random_vs, random_acs)
            ][:-1]
        )

        # Act
        x0, x_init, u_init = ws.generate(initial_state, reference_trajectory)
        x_init = np.array(x_init)
        u_init = np.array(u_init)

        # Assert
        # Check shapes
        self.assertEqual(x_init.shape, expected_x_init.shape)
        self.assertEqual(u_init.shape, expected_u_init.shape)

        # Check values (assuming `generate` would use these random inputs)
        np.testing.assert_array_equal(x0, expected_x0)
        np.testing.assert_array_equal(x_init, expected_x_init)
        np.testing.assert_array_equal(u_init, expected_u_init)


if __name__ == "__main__":
    unittest.main()
