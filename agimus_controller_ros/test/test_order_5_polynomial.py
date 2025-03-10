import numpy as np
import unittest
import rclpy
from agimus_controller_ros.simple_trajectory_publisher import QuinticTrajectory


class TestQunintTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()
        return super().tearDownClass()

    def test_quintic_trajectory(self):
        amp = 0.5
        scale_duration = 0.7
        obj = QuinticTrajectory(scale_duration=scale_duration, amp=amp)
        times = np.linspace(0, obj.scale_duration, 200)
        positions = [obj.get_value_at_t(t)[0] for t in times]

        for position in positions:
            self.assertGreaterEqual(position, 0)
            self.assertLessEqual(position, obj.amp)

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
