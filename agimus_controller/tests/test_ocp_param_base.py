import unittest
import numpy as np

<<<<<<< Updated upstream
from agimus_controller.agimus_controller.ocp_param_base import OCPParamsCrocoBase
from agimus_controller.agimus_controller.trajectory import (
    WeightedTrajectoryPoint,
    TrajectoryPoint,
    TrajectoryPointWeights,
)

=======
from agimus_controller.ocp_param_base import OCPParamsCrocoBase
from agimus_controller.trajectory import WeightedTrajectoryPoint, TrajectoryPoint, TrajectoryPointWeights
>>>>>>> Stashed changes

class TestOCPParamsCrocoBase(unittest.TestCase):
    """Test the OCPParamsCrocoBase class."""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_initialization(self):
        dt = np.float64(0.01)
        T = 100
        solver_iters = 50
        point = TrajectoryPoint(
            robot_configuration=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        weights = TrajectoryPointWeights(
            w_robot_configuration=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        weighted_trajectory_point = WeightedTrajectoryPoint(point, weights)
        weighted_trajectory_points = [weighted_trajectory_point]
        armature = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ee_name = "end_effector"
        p_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        params = OCPParamsCrocoBase(
            dt=dt,
            T=T,
            solver_iters=solver_iters,
            WeightedTrajectoryPoints=weighted_trajectory_points,
            armature=armature,
            ee_name=ee_name,
        )

        self.assertEqual(params.dt, dt)
        self.assertEqual(params.T, T)
        self.assertEqual(params.qp_iters, 200)
        self.assertEqual(params.solver_iters, solver_iters)
        self.assertEqual(params.termination_tolerance, 1e-3)
        self.assertEqual(params.eps_abs, 1e-6)
        self.assertEqual(params.eps_rel, 0)
        self.assertFalse(params.callbacks)
        self.assertEqual(params.WeightedTrajectoryPoints, weighted_trajectory_points)
        self.assertTrue(np.array_equal(params.armature, armature))
        self.assertEqual(params.ee_name, ee_name)


if __name__ == "__main__":
    unittest.main()
