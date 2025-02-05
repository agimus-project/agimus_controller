import unittest
import numpy as np
from pathlib import Path
import example_robot_data as robex
import pinocchio as pin

from agimus_controller.ocp.ocp_croco_goal_reaching import OCPCrocoGoalReaching
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class TestOCPGoalReaching(unittest.TestCase):
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

        # OCP parameters
        dt = 0.05
        horizon_size = 200
        solver_iters = 100
        callbacks = False

        self._ocp_params = OCPParamsBaseCroco(
            dt=dt,
            horizon_size=horizon_size,
            dt_factor_n_seq=[(1, horizon_size - 1)],
            solver_iters=solver_iters,
            callbacks=callbacks,
        )

    def test_ocp_solution(self):
        # Set initial state
        q0 = pin.neutral(self.robot_models.robot_model)
        state_warmstart = []
        control_warmstart = []
        trajectory_points = []

        ee_pose = pin.SE3(np.eye(3), np.array([0.5, 0.2, 0.5]))
        for i in range(1, self._ocp_params.horizon_size):
            u_ref = np.zeros(self.robot_models.robot_model.nv)
            trajectory_points.append(
                WeightedTrajectoryPoint(
                    TrajectoryPoint(
                        robot_configuration=q0,
                        robot_velocity=np.zeros(self.robot_models.robot_model.nv),
                        robot_effort=u_ref,
                        end_effector_poses={"panda_hand_tcp": ee_pose},
                    ),
                    TrajectoryPointWeights(
                        w_robot_configuration=0.01
                        * np.ones(self.robot_models.robot_model.nq),
                        w_robot_velocity=0.01
                        * np.ones(self.robot_models.robot_model.nv),
                        w_robot_effort=0.0001
                        * np.ones(self.robot_models.robot_model.nv),
                        w_end_effector_poses={
                            "panda_hand_tcp": (
                                1e3 * np.ones(6)
                                if i < self._ocp_params.horizon_size - 1
                                else 1e3 * np.ones(6)
                            )
                        },
                    ),
                )
            )
            state_warmstart.append(
                np.concatenate((q0, np.zeros(self.robot_models.robot_model.nv)))
            )
            control_warmstart.append(u_ref)

        # Solve OCP
        self._state_reg = np.concatenate(
            (q0, np.zeros(self.robot_models.robot_model.nv))
        )
        self._ocp = OCPCrocoGoalReaching(self.robot_models, self._ocp_params)
        self._ocp.set_reference_weighted_trajectory(trajectory_points)
        self._ocp.solve(self._state_reg, state_warmstart, control_warmstart)

        data = self.robot_models.robot_model.createData()
        pin.framesForwardKinematics(
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
        # self._visualize() # For debug purposes

    def _visualize(self):
        vis = self._create_viewer()
        for xs in self._ocp.ocp_results.states:
            q = xs[: self.robot_models.robot_model.nq]
            vis.display(q)
            input()

    def _create_viewer(self):
        import meshcat
        from pinocchio import visualize

        viz = visualize.MeshcatVisualizer(
            model=self.robot_models.robot_model,
            collision_model=self.robot_models.collision_model,
            visual_model=self.robot_models.visual_model,
        )
        viz.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
        viz.loadViewerModel("pinocchio")

        viz.displayCollisions(True)
        return viz


if __name__ == "__main__":
    unittest.main()
