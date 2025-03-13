import numpy as np
from pathlib import Path
import example_robot_data
import unittest
import crocoddyl
import pinocchio

from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)

from agimus_controller.ocp.ocp_croco_generic import BuildData
from agimus_controller.ocp import ocp_croco_generic
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_base import DTFactorsNSeq
from agimus_controller.ocp_param_base import OCPParamsBaseCroco


class OCPCrocoGenericBuilderTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._rmodel = pinocchio.buildSampleModelManipulator()
        self._rgmodel = pinocchio.buildSampleGeometryModelManipulator(self._rmodel)
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)
        self._build_data = BuildData(self._state, self._actuation, self._rgmodel)

    def test_residual_model_state(self):
        xref = np.zeros(self._state.nx)
        differential = ocp_croco_generic.DifferentialActionModelFreeFwdDynamics(
            costs=[
                ocp_croco_generic.CostModelSumItem(
                    "state",
                    ocp_croco_generic.CostModelResidual(
                        residual=ocp_croco_generic.ResidualModelState(xref),
                        activation=ocp_croco_generic.ActivationModelWeightedQuad(1.0),
                    ),
                    update=True,
                )
            ]
        )
        cmodel = differential.build(self._build_data)
        cdata = cmodel.createData()
        x = np.random.random(self._state.nx)

        cmodel.calc(cdata, x)
        np.testing.assert_array_equal(cdata.costs.costs["state"].residual.r, x - xref)
        self.assertAlmostEqual(
            cdata.costs.costs["state"].cost, np.sum(0.5 * (x - xref) ** 2)
        )

        xref = np.random.random(self._state.nx)
        pt = WeightedTrajectoryPoint(
            point=TrajectoryPoint(
                robot_configuration=xref[: self._rmodel.nq],
                robot_velocity=xref[self._rmodel.nq :],
            ),
            weights=TrajectoryPointWeights(
                w_robot_configuration=0.5 * np.ones(self._rmodel.nv),
                w_robot_velocity=10 * np.ones(self._rmodel.nv),
            ),
        )
        w = pt.weights.w_robot_state

        differential.update(self._build_data, cmodel, pt)
        cmodel.calc(cdata, x)
        np.testing.assert_array_equal(cdata.costs.costs["state"].residual.r, x - xref)
        self.assertAlmostEqual(
            cdata.costs.costs["state"].cost, np.sum(0.5 * w * (x - xref) ** 2)
        )

    def test_residual_model_control(self):
        uref = np.zeros(self._actuation.nu)
        differential = ocp_croco_generic.DifferentialActionModelFreeFwdDynamics(
            costs=[
                ocp_croco_generic.CostModelSumItem(
                    "control",
                    ocp_croco_generic.CostModelResidual(
                        residual=ocp_croco_generic.ResidualModelControl(uref),
                        activation=ocp_croco_generic.ActivationModelWeightedQuad(1.0),
                    ),
                    update=True,
                )
            ]
        )
        cmodel = differential.build(self._build_data)
        cdata = cmodel.createData()
        x = np.random.random(self._state.nx)
        u = np.random.random(self._actuation.nu)

        cmodel.calc(cdata, x, u)
        np.testing.assert_array_equal(cdata.costs.costs["control"].residual.r, u - uref)
        self.assertAlmostEqual(
            cdata.costs.costs["control"].cost, np.sum(0.5 * (u - uref) ** 2)
        )

        uref = np.random.random(self._actuation.nu)
        pt = WeightedTrajectoryPoint(
            point=TrajectoryPoint(robot_effort=uref),
            weights=TrajectoryPointWeights(
                w_robot_effort=0.5 * np.ones(self._rmodel.nv)
            ),
        )
        w = pt.weights.w_robot_effort

        differential.update(self._build_data, cmodel, pt)
        cmodel.calc(cdata, x, u)
        np.testing.assert_array_equal(cdata.costs.costs["control"].residual.r, u - uref)
        self.assertAlmostEqual(
            cdata.costs.costs["control"].cost, np.sum(0.5 * w * (u - uref) ** 2)
        )


class OCPCrocoGenericTest(unittest.TestCase):
    def setUp(self):
        ### LOAD ROBOT
        robot = example_robot_data.load("panda")
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
            robot_urdf=urdf_path,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=False,
            armature=armature,
        )

        self.robot_models = RobotModels(self.params)

        # OCP parameters
        dt = 0.05
        horizon_size = 200
        solver_iters = 100
        callbacks = False

        self._ocp_params = OCPParamsBaseCroco(
            dt=dt,
            horizon_size=horizon_size,
            dt_factor_n_seq=DTFactorsNSeq(factors=[1], n_steps=[horizon_size]),
            solver_iters=solver_iters,
            callbacks=callbacks,
        )

    def test_ocp_solution(self):
        # Set initial state
        q0 = pinocchio.neutral(self.robot_models.robot_model)
        ee_pose = pinocchio.SE3(np.eye(3), np.array([0.5, 0.2, 0.5]))

        n_states = self._ocp_params.n_controls + 1

        state_warmstart = [
            np.concatenate((q0, np.zeros(self.robot_models.robot_model.nv)))
        ] * n_states
        control_warmstart = [
            np.zeros(self.robot_models.robot_model.nv)
        ] * self._ocp_params.n_controls
        trajectory_points = [
            WeightedTrajectoryPoint(
                TrajectoryPoint(
                    robot_configuration=q0,
                    robot_velocity=np.zeros(self.robot_models.robot_model.nv),
                    robot_effort=np.zeros(self.robot_models.robot_model.nv),
                    end_effector_poses={"panda_hand_tcp": ee_pose},
                ),
                TrajectoryPointWeights(
                    w_robot_configuration=0.01
                    * np.ones(self.robot_models.robot_model.nq),
                    w_robot_velocity=0.01 * np.ones(self.robot_models.robot_model.nv),
                    w_robot_effort=0.0001 * np.ones(self.robot_models.robot_model.nv),
                    w_end_effector_poses={"panda_hand_tcp": (1e3 * np.ones(6))},
                ),
            )
        ] * n_states

        # Solve OCP
        self._state_reg = np.concatenate(
            (q0, np.zeros(self.robot_models.robot_model.nv))
        )
        ocp_definition_file = ocp_croco_generic.OCPCrocoGeneric.get_default_yaml_file(
            "ocp_goal_reaching.yaml"
        )
        self._ocp = ocp_croco_generic.OCPCrocoGeneric(
            self.robot_models, self._ocp_params, yaml_file=ocp_definition_file
        )
        self._ocp.set_reference_weighted_trajectory(trajectory_points)
        self._ocp.solve(self._state_reg, state_warmstart, control_warmstart)

        data = self.robot_models.robot_model.createData()
        pinocchio.framesForwardKinematics(
            self.robot_models.robot_model,
            data,
            self._ocp.ocp_results.states[-1][: self.robot_models.robot_model.nq],
        )
        # Test that the last position of the end-effector is close to the target
        self.assertAlmostEqual(
            np.linalg.norm(
                data.oMf[
                    self.robot_models.robot_model.getFrameId("panda_hand_tcp")
                ].translation
                - ee_pose.translation
            ),
            0.0,
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
