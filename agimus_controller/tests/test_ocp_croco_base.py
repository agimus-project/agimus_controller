import unittest
from unittest.mock import MagicMock, patch

import example_robot_data as robex
import numpy as np

from agimus_controller.ocp_base_croco import OCPBaseCroco
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModelFactory


class TestOCPBaseCroco(unittest.TestCase):
    def setUp(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        self.mock_robot_model_factory = RobotModelFactory()

        mock_robot = robex.load("panda")
        mock_robot_model = mock_robot.model
        mock_collision_model = mock_robot.collision_model
        mock_armature = np.array([])

        self.mock_robot_model_factory._rmodel = mock_robot_model
        self.mock_robot_model_factory._complete_collision_model = mock_collision_model
        self.mock_robot_model_factory._armature = mock_armature

        self.mock_ocp_params = MagicMock(spec=OCPParamsBaseCroco)

        # Set mock parameters
        self.mock_ocp_params.T = 10
        self.mock_ocp_params.dt = 0.1
        self.mock_ocp_params.use_filter_line_search = True
        self.mock_ocp_params.termination_tolerance = 1e-6
        self.mock_ocp_params.qp_iters = 10
        self.mock_ocp_params.eps_abs = 1e-8
        self.mock_ocp_params.eps_rel = 1e-6
        self.mock_ocp_params.callbacks = True
        self.mock_ocp_params.solver_iters = 100

        # Create a concrete implementation of OCPBaseCroco
        class TestOCPCroco(OCPBaseCroco):
            @property
            def runningModelList(self):
                return None

            @property
            def terminalModel(self):
                return None

            def set_reference_horizon(self, horizon_size):
                return None

        self.ocp = TestOCPCroco(self.mock_robot_model_factory, self.mock_ocp_params)

    def test_horizon_size(self):
        """Test the horizon_size property."""
        self.assertEqual(self.ocp.horizon_size, self.mock_ocp_params.T)

    def test_dt(self):
        """Test the dt property."""
        self.assertAlmostEqual(self.ocp.dt, self.mock_ocp_params.dt)


class TestSimpleOCPCroco(unittest.TestCase):
    def setUp(self):
        # Mock the RobotModelFactory and OCPParamsCrocoBase
        self.robot_model_factory = RobotModelFactory()

        robot = robex.load("panda")
        robot_model = robot.model
        collision_model = robot.collision_model
        armature = np.array([])

        self.robot_model_factory._rmodel = robot_model
        self.robot_model_factory._complete_collision_model = collision_model
        self.robot_model_factory.armature = armature

        self.ocp_params = OCPParamsBaseCroco()

        # Set mock parameters
        self.ocp_params.T = 10
        self.ocp_params.dt = 0.1
        self.ocp_params.use_filter_line_search = True
        self.ocp_params.termination_tolerance = 1e-6
        self.ocp_params.qp_iters = 10
        self.ocp_params.eps_abs = 1e-8
        self.ocp_params.eps_rel = 1e-6
        self.ocp_params.callbacks = True
        self.ocp_params.solver_iters = 100

        # Create a concrete implementation of OCPBaseCroco
        class TestOCPCroco(OCPBaseCroco):
            @property
            def runningModelList(self):
                return None

            @property
            def terminalModel(self):
                return None

            def set_reference_horizon(self, horizon_size):
                ### Not implemented in this OCP example.
                return None

        self.ocp = TestOCPCroco(self.robot_model_factory, self.ocp_params)


if __name__ == "__main__":
    unittest.main()
