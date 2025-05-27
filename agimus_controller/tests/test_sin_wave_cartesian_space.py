from copy import deepcopy
import example_robot_data as robex
import numpy as np
from pathlib import Path
import unittest

from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels
from agimus_controller.trajectories.sine_wave_params import SinWaveParams
from agimus_controller.trajectories.sine_wave_cartesian_space import (
    SinusWaveCartesianSpace,
)

VISUALIZE = False
if VISUALIZE:
    import matplotlib.pyplot as plt


class TestSinWaveCartesianTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method sets up the shared environment for all test cases in the class.
        """
        # Load the example robot model using example robot data to get the URDF path.
        robot = robex.load("panda")
        urdf_path = Path(robot.urdf)
        env_urdf = Path(__file__).parent / "resources" / "environment.xacro"
        srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
        urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
        free_flyer = False
        q0 = np.array(
            [
                -0.3619834760502907,
                -1.3575006398318104,
                0.969610481368033,
                -2.6028532848927295,
                0.2040785081450368,
                1.9436352693107668,
                0.6423896937386857,
                0.0,
                0.0,
            ]
        )
        moving_joint_names = [f"panda_joint{x}" for x in range(1, 8)]
        reduced_nq = len(moving_joint_names)
        cls.params = RobotModelParameters(
            q0=q0,
            free_flyer=free_flyer,
            moving_joint_names=moving_joint_names,
            robot_urdf=urdf_path,
            env_urdf=env_urdf,
            srdf=srdf_path,
            urdf_meshes_dir=urdf_meshes_dir,
            collision_as_capsule=True,
            self_collision=True,
            armature=np.linspace(0.1, 0.9, reduced_nq),
        )
        cls.sine_wave_params = SinWaveParams(
            amplitude=np.array([0.1, 0.1, 0.0]),
            period=np.array([4.0, 4.0, 4.0]),
            scale_duration=np.array([0.2, 0.2, 0.2]),
        )

    def setUp(self):
        """
        This method ensures that a fresh RobotModelParameters and RobotModels instance
        are created for each test case.
        """
        self.params = deepcopy(self.params)
        self.robot_models = RobotModels(self.params)

    def plot(self, times, traj, obj):
        if VISUALIZE:
            traj_ee_ik = np.array(
                [
                    obj.get_end_effector_pose_from_q(
                        traj_point.point.robot_configuration
                    )
                    for traj_point in traj
                ]
            )
            traj_ee = np.array(
                [
                    traj_point.point.end_effector_poses["panda_hand_tcp"]
                    for traj_point in traj
                ]
            )
            traj_q = np.array(
                [traj_point.point.robot_configuration for traj_point in traj]
            )
            traj_dq = np.array([traj_point.point.robot_velocity for traj_point in traj])
            plt.figure()
            plt.xlabel("Time (s)")
            plt.suptitle("Sine wave cartesian space trajectory")
            plt.grid()
            ax = plt.subplot(2, 2, 1)
            ax.set_title("End effector trajectory position")
            ax.plot(
                times,
                np.hstack([traj_ee[:, :3], traj_ee_ik[:, :3]]),
                label=[
                    "des ee x",
                    "des ee y",
                    "des ee z",
                    "ik ee x",
                    "ik ee y",
                    "ik ee z",
                ],
            )
            ax = plt.subplot(2, 2, 2)
            ax.set_title("Robot configuration trajectory")
            ax.plot(times, traj_q)
            ax = plt.subplot(2, 2, 3)
            ax.set_title("Robot velocity trajectory")
            ax.plot(times, traj_dq)
            ax = plt.subplot(2, 2, 4)
            ax.set_title("End effector trajectory quaternion")
            ax.plot(
                times,
                np.hstack([traj_ee[:, 3:], traj_ee_ik[:, 3:]]),
                label=[
                    "des ee qx",
                    "des ee qy",
                    "des ee qz",
                    "des ee qw",
                    "ik ee qx",
                    "ik ee qy",
                    "ik ee qz",
                    "ik ee qw",
                ],
            )
            plt.legend()
            plt.show()

    def test_sin_wave_cartesian_space_trajectory(self):
        obj = SinusWaveCartesianSpace(
            sine_wave_params=self.sine_wave_params,
            w_q=np.array([1.0]),
            w_qdot=np.array([0.1]),
            w_qddot=np.array([0.000001]),
            w_robot_effort=np.array([0.0003]),
            w_pose=np.array([0.1]),
            ee_frame_name="panda_hand_tcp",
            mask=[True, True, True, True, True, True],
        )
        obj.initialize(self.robot_models.robot_model, self.params.q0[:7])

        dt = 1e-1
        duration = np.max(self.sine_wave_params.scale_duration) + 2 * np.max(
            self.sine_wave_params.period
        )
        times = np.linspace(0, duration, int(duration / dt))
        traj = [obj.get_traj_point_at_t(t) for t in times]
        self.plot(times, traj, obj)
        for traj_point in traj:
            ik_ee_pos = obj.get_end_effector_pose_from_q(
                traj_point.point.robot_configuration
            )
            ee_pos = traj_point.point.end_effector_poses["panda_hand_tcp"].copy()
            np.testing.assert_allclose(ik_ee_pos, ee_pos, atol=1e-3)

    def test_sin_wave_cartesian_space_trajectory_3d(self):
        obj = SinusWaveCartesianSpace(
            sine_wave_params=self.sine_wave_params,
            w_q=np.array([1.0]),
            w_qdot=np.array([0.1]),
            w_qddot=np.array([0.000001]),
            w_robot_effort=np.array([0.0003]),
            w_pose=np.array([0.1]),
            ee_frame_name="panda_hand_tcp",
            mask=[True, True, True, False, False, False],
        )
        obj.optimize_orientation = False
        obj.initialize(self.robot_models.robot_model, self.params.q0[:7])

        dt = 1e-1
        duration = np.max(self.sine_wave_params.scale_duration) + 2 * np.max(
            self.sine_wave_params.period
        )
        times = np.linspace(0, duration, int(duration / dt))
        traj = [obj.get_traj_point_at_t(t) for t in times]
        self.plot(times, traj, obj)
        for traj_point in traj:
            ik_ee_pos = obj.get_end_effector_pose_from_q(
                traj_point.point.robot_configuration
            )
            ee_pos = traj_point.point.end_effector_poses["panda_hand_tcp"].copy()
            np.testing.assert_allclose(ik_ee_pos[:3], ee_pos[:3], atol=1e-3)

    def test_ik_6D(self):
        obj = SinusWaveCartesianSpace(
            sine_wave_params=self.sine_wave_params,
            w_q=np.array([1.0]),
            w_qdot=np.array([0.1]),
            w_qddot=np.array([0.000001]),
            w_robot_effort=np.array([0.0003]),
            w_pose=np.array([0.1]),
            ee_frame_name="panda_hand_tcp",
            mask=[True, True, True, True, True, True],
        )
        obj.initialize(
            self.robot_models.robot_model, self.params.q0[:7] + np.array(7 * [0.1])
        )
        ee_pos = obj.get_end_effector_pose_from_q_as_se3(self.params.q0[:7])
        ik_q, ik_dq = obj.inverse_kinematics(
            ee_pos,
            np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
            precision=1e-4,
        )
        ik_ee_pos = obj.get_end_effector_pose_from_q_as_se3(ik_q)
        np.testing.assert_allclose(
            ik_dq,
            -np.array(
                [0.640289, -0.419278, 0.146452, -1.156815, 0.21497, 0.43003, 0.108381]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(ik_ee_pos.homogeneous, ee_pos.homogeneous, atol=1e-3)

    def test_ik_3D(self):
        obj = SinusWaveCartesianSpace(
            sine_wave_params=self.sine_wave_params,
            w_q=np.array([1.0]),
            w_qdot=np.array([0.1]),
            w_qddot=np.array([0.000001]),
            w_robot_effort=np.array([0.0003]),
            w_pose=np.array([0.1]),
            ee_frame_name="panda_hand_tcp",
            mask=[True, True, True, False, False, False],
        )
        obj.initialize(
            self.robot_models.robot_model, self.params.q0[:7] + np.array(7 * [0.1])
        )
        ee_pos = obj.get_end_effector_pose_from_q_as_se3(self.params.q0[:7])
        ik_q, ik_dq = obj.inverse_kinematics(ee_pos, np.zeros(6))
        ik_ee_pos = obj.get_end_effector_pose_from_q_as_se3(ik_q)
        np.testing.assert_allclose(ik_dq, np.zeros(7), atol=1e-3)
        np.testing.assert_allclose(ik_ee_pos.translation, ee_pos.translation, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
