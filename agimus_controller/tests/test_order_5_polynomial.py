import numpy as np
import unittest
from agimus_controller.trajectories.quintic_trajectory import QuinticTrajectory


class TestQuinticTrajectory(unittest.TestCase):
    def test_quintic_trajectory(self):
        N = 3
        scale_duration = N * [0.7]
        obj = QuinticTrajectory(scale_duration=scale_duration)
        dt = 1e-5
        times = np.linspace(0, obj.scale_duration[0], int(scale_duration[0] / dt))
        positions = [obj.get_value_at_t(t) for t in times]

        for idx, position in enumerate(positions):
            polynom, d_polynom, dd_polynom = position
            np.testing.assert_array_less(np.zeros(N), polynom)

            # test derivatives by finite difference
            if idx < len(positions) - 1:
                next_position = positions[idx + 1]
                next_polynom, d_next_polynom, dd_next_polynom = next_position
                np.testing.assert_almost_equal(
                    (next_polynom - polynom) / dt,
                    d_next_polynom,
                )
                np.testing.assert_almost_equal(
                    (d_next_polynom - d_polynom) / dt,
                    dd_next_polynom,
                )

        if False:
            import matplotlib.pyplot as plt

            plt.plot(times, positions)
            plt.xlabel("Time (s)")
            plt.ylabel("Position")
            plt.title("Quintic Trajectory from 0 to 1")
            plt.grid()
            plt.show()


if __name__ == "__main__":
    unittest.main()
