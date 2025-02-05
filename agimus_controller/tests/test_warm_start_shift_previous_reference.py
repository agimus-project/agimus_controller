from pathlib import Path
import unittest

import numpy as np
import example_robot_data as robex

from agimus_controller.mpc_data import OCPResults
from agimus_controller.warm_start_shift_previous_solution import (
    WarmStartShiftPreviousSolution,
)
from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.ocp.ocp_croco_goal_reaching import OCPCrocoGoalReaching


class TestWarmStart(unittest.TestCase):
    def setUp(self):
        ### LOAD ROBOT
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        reduced_nq = robot.model.nq - len(locked_joint_names)
        moving_joint_names = set(robot.model.names) - set(
            locked_joint_names + ["universe"]
        )
        q0 = np.zeros(robot.model.nq)
        armature = np.full(reduced_nq, 0.1)

        # Store shared initial parameters
        self.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            moving_joint_names=moving_joint_names,
            urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=False,
            armature=armature,
        )

        self.robot_models = RobotModels(self.params)

    def test_initialization(self):
        ws = WarmStartShiftPreviousSolution()

        with self.assertRaises(AssertionError):
            ws.generate(None, None)

    def test_generate(self):
        dt = 0.1
        ws = WarmStartShiftPreviousSolution()
        ocp_params = OCPParamsBaseCroco(
            dt=0.1,
            solver_iters=1000,
            horizon_size=4,
            dt_factor_n_seq=[
                (1, 2),
                (2, 1),
            ],
        )
        assert ocp_params.timesteps == (0.1, 0.1, 0.2)
        nu = 3

        ws.setup(
            self.robot_models,
            ocp_params,
        )

        ocp = OCPCrocoGoalReaching(self.robot_models, ocp_params)
        model = self.robot_models.robot_model
        controls = [np.random.random(model.nv) for _ in range(nu)]
        ocp.problem.x0 = np.zeros(model.nq + model.nv)
        states = ocp.problem.rollout(controls)
        # Sanity check
        for i in range(nu):
            m = ocp.problem.runningModels[i]
            d = ocp.problem.runningDatas[i]
            m.calc(d, states[i], controls[i])
            np.testing.assert_array_equal(states[i + 1], d.xnext)

        prev_solution = OCPResults(
            states=states.copy(),
            ricatti_gains=[],
            feed_forward_terms=controls.copy(),
        )

        ws.update_previous_solution(prev_solution)

        init_state = TrajectoryPoint(
            robot_configuration=-10 * np.ones(model.nq),
            robot_velocity=100 * np.ones(model.nv),
        )
        x0, x_init, u_init = ws.generate(init_state, [])

        # Assert
        # Check shapes
        assert len(x_init) == nu
        assert len(u_init) == nu

        # Check values (assuming `generate` would use these random inputs)
        np.testing.assert_array_equal(x0[: model.nq], init_state.robot_configuration)
        np.testing.assert_array_equal(x0[model.nq :], init_state.robot_velocity)
        # Integrate each point and compare to x_init.
        # We skip the first control because this is not in x_init (discarded).
        for i in range(1, nu):
            m = ocp.problem.runningModels[i]
            d = ocp.problem.runningDatas[i]
            m.dt = ocp_params.dt
            m.calc(d, states[i], controls[i])

            # This assert is not based on something we actually desire.
            # We may want something smarted for the control
            np.testing.assert_array_equal(u_init[i - 1], controls[i])
            np.testing.assert_array_equal(x_init[i - 1], d.xnext)


if __name__ == "__main__":
    unittest.main()
