import unittest

import numpy as np
import example_robot_data as robex

from agimus_controller.ocp_base_croco import OCPCrocoBase
from agimus_controller.ocp_param_base import OCPParamsCrocoBase


class TestOCPCrocoBase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        pass

    def setUp(self):
        """Set up the test environment."""
        # Load the robot
        self.robot = robex.load('example_robot_description')
        self.rmodel = self.robot.model
        self.cmodel = self.robot.collision_model

        # Set some fixed params for unit testing
        self.dt = 0.01
        self.T = 100
        self.qp_iters = 200
        self.solve_iters = 200
        self.callbacks = True
        self.x0 = np.zeros(self.rmodel.nq + self.rmodel.nv)
        self.OCPParams = OCPParamsCrocoBase(
            dt = self.dt,
            T = self.T,
            qp_iters=self.qp_iters,
            solver_iters=self.solve_iters,
            callbacks=self.callbacks,
            ...,
        )
        return super().setUp()

    def test_abstract_class_instantiation(self):
        """Test the instantiation of the OCPCrocoBase class."""
        with self.assertRaises(TypeError):
            OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)

    def test_horizon_size(self):
        """Test the horizon_size property of the OCPCrocoBase class."""
        ocp = OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)
        self.assertEqual(ocp.horizon_size, self.T)

    def test_dt(self):
        """Test the dt property of the OCPCrocoBase class."""
        ocp = OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)
        self.assertEqual(ocp.dt, self.dt)

    def test_x0(self):
        """Test the x0 property of the OCPCrocoBase class."""
        ocp = OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)
        self.assertTrue(np.array_equal(ocp.x0, self.x0))

    def test_x_init(self):
        """Test the x_init method of the OCPCrocoBase class."""
        ocp = OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)
        x_init = ocp.x_init()
        self.assertEqual(len(x_init), self.T)
        self.assertTrue(np.array_equal(x_init[0], self.x0))

    def test_u_init(self):
        """Test the u_init method of the OCPCrocoBase class."""
        ocp = OCPCrocoBase(self.rmodel, self.cmodel, self.OCPParams)
        u_init = ocp.u_init()
        self.assertEqual(len(u_init), self.T - 1)

    # def test_solve(self):

if __name__ == '__main__':
    unittest.main()
