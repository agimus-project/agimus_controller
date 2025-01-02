import unittest
import numpy as np

from agimus_controller.ocp_param_base import OCPParamsBaseCroco


class TestOCPParamsCrocoBase(unittest.TestCase):
    """
    TestOCPParamsCrocoBase is a unit test class for testing the OCPParamsBaseCroco class.

    Methods:
        __init__(methodName="runTest"): Initializes the test case instance.
        test_initialization(): Tests the initialization of the OCPParamsBaseCroco class with various parameters.
    """

    """Test the OCPParamsBaseCroco class."""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_initialization(self):
        """
        Test the initialization of the OCPParamsBaseCroco class.

        This test verifies that the parameters passed to the OCPParamsBaseCroco
        constructor are correctly assigned to the instance attributes.

        Test Parameters:
        - dt (float): Time step for the OCP solver.
        - T (int): Total number of time steps.
        - solver_iters (int): Number of solver iterations.
        - qp_iters (int): Number of QP iterations.
        - termination_tolerance (float): Tolerance for termination criteria.
        - eps_abs (float): Absolute tolerance for the solver.
        - eps_rel (float): Relative tolerance for the solver.
        - callbacks (bool): Flag to enable or disable callbacks.

        Assertions:
        - Asserts that the instance attribute `dt` is equal to the input `dt`.
        - Asserts that the instance attribute `T` is equal to the input `T`.
        - Asserts that the instance attribute `qp_iters` is equal to the input `qp_iters`.
        - Asserts that the instance attribute `solver_iters` is equal to the input `solver_iters`.
        - Asserts that the instance attribute `termination_tolerance` is equal to the input `termination_tolerance`.
        - Asserts that the instance attribute `eps_abs` is equal to the input `eps_abs`.
        - Asserts that the instance attribute `eps_rel` is equal to the input `eps_rel`.
        - Asserts that the instance attribute `callbacks` is False.
        """
        dt = np.float64(0.01)
        T = 100
        solver_iters = 50
        qp_iters = 200
        termination_tolerance = 1e-3
        eps_abs = 1e-6
        eps_rel = 0
        callbacks = False
        params = OCPParamsBaseCroco(
            dt=dt,
            T=T,
            solver_iters=solver_iters,
            qp_iters=qp_iters,
            termination_tolerance=termination_tolerance,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            callbacks=callbacks,
        )
        self.assertEqual(params.dt, dt)
        self.assertEqual(params.T, T)
        self.assertEqual(params.qp_iters, qp_iters)
        self.assertEqual(params.solver_iters, solver_iters)
        self.assertEqual(params.termination_tolerance, termination_tolerance)
        self.assertEqual(params.eps_abs, eps_abs)
        self.assertEqual(params.eps_rel, eps_rel)
        self.assertFalse(params.callbacks)


if __name__ == "__main__":
    unittest.main()
