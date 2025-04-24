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

    def setUp(self):
        """
        This method ensures that a fresh RobotModelParameters and RobotModels instance
        are created for each test case.
        """
        self.params = deepcopy(self.params)
        self.robot_models = RobotModels(self.params)
        self.visualize = False

    def test_sin_wave_cartesian_space_trajectory(self):
        sine_wave_params = SinWaveParams(
            amplitude=0.1,
            period=2.0,
            scale_duration=0.2,
        )
        obj = SinusWaveCartesianSpace(
            sine_wave_params=sine_wave_params,
            w_q=np.array([1.0]),
            w_qdot=np.array([0.1]),
            w_qddot=np.array([0.000001]),
            w_robot_effort=np.array([0.0003]),
            w_pose=np.array([0.1]),
            ee_frame_name="panda_hand_tcp",
        )
        obj.set_pin_model(self.robot_models.robot_model)
        obj.set_init_configuration(self.params.q0[:7])

        dt = 1e-1
        duration = sine_wave_params.scale_duration + 2 * sine_wave_params.period
        times = np.linspace(0, duration, int(duration / dt))
        traj = [obj.get_traj_point_at_t(t) for t in times]
        # for idx, traj_point in enumerate(traj):
        #     # TODO do some testing here.
        #     pass

        if self.visualize:
            traj_ee = np.array(
                [
                    traj_point.point.end_effector_poses["panda_hand_tcp"]
                    for traj_point in traj
                ]
            )
            traj_q = np.array(
                [traj_point.point.robot_configuration for traj_point in traj]
            )
            traj_v = np.array([traj_point.point.robot_velocity for traj_point in traj])
            traj_a = np.array(
                [traj_point.point.robot_acceleration for traj_point in traj]
            )

            import matplotlib.pyplot as plt

            plt.figure()
            plt.xlabel("Time (s)")
            plt.title("Sine wave cartesian space trajectory")
            plt.grid()
            ax = plt.subplot(2, 2, 1)
            ax.plot(times, traj_ee)
            ax = plt.subplot(2, 2, 2)
            ax.plot(times, traj_q)
            ax = plt.subplot(2, 2, 3)
            ax.plot(times, traj_v)
            ax = plt.subplot(2, 2, 4)
            ax.plot(times, traj_a)
            plt.show()


if __name__ == "__main__":
    unittest.main()
