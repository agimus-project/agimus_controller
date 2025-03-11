import numpy as np
import unittest
import rclpy
from agimus_controller_ros.simple_trajectory_publisher import QuinticTrajectory


class TestQuinticTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()
        return super().tearDownClass()

    def test_quintic_trajectory(self):
        amp = 1.5
        scale_duration = 0.7
        obj = QuinticTrajectory(scale_duration=scale_duration, amp=amp)
        dt = 1e-5
        precision = 1e-3  # precision for finite difference
        times = np.linspace(0, obj.scale_duration, int(scale_duration / dt))
        positions = [obj.get_value_at_t(t) for t in times]

        for idx, position in enumerate(positions):
            polynom, d_polynom, dd_polynom = position
            self.assertGreaterEqual(polynom, 0)
            self.assertLessEqual(polynom, obj.amp)

            # test derivatives by finite difference
            if idx < len(positions) - 1:
                next_position = positions[idx + 1]
                next_polynom, d_next_polynom, dd_next_polynom = next_position
                self.assertLessEqual(
                    np.abs((next_polynom - polynom) / dt - d_next_polynom),
                    precision,
                )
                self.assertLessEqual(
                    np.abs((d_next_polynom - d_polynom) / dt - dd_next_polynom),
                    precision,
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
