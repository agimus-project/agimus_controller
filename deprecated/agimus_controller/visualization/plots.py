import time
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from agimus_controller.visualization.wrapper_meshcat import MeshcatWrapper
from agimus_controller.utils.process_handler import MeshcatServer
from agimus_controller.utils.pin_utils import (
    get_ee_pose_from_configuration,
    get_last_joint,
)


class MPCPlots:
    def __init__(
        self,
        croco_xs,
        croco_us,
        whole_x_plan,
        whole_u_plan,
        rmodel,
        vmodel,
        cmodel,
        DT,
        ee_frame_name: str,
        viewer=None,
    ):
        if viewer is not None:
            self.viewer = viewer
        self.DT = DT
        self.rmodel = rmodel
        self.vmodel = vmodel
        self.cmodel = cmodel
        self._rdata = self.rmodel.createData()

        (
            self._last_joint_name,
            self._last_joint_id,
            self._last_joint_frame_id,
        ) = get_last_joint(self.rmodel)

        self.nq = self.rmodel.nq
        self.croco_xs = croco_xs
        self.croco_us = croco_us
        self.whole_x_plan = whole_x_plan
        self.whole_u_plan = whole_u_plan
        self.path_length = (whole_x_plan.shape[0] - 1) * self.DT

        self._ee_frame_name = ee_frame_name
        self._id_ee_frame_name = self.rmodel.getFrameId(self._ee_frame_name)
        self.meshcat_server = MeshcatServer()
        self.meshcat_server.start()
        self.MeshcatVis = MeshcatWrapper()
        self.vis = self.MeshcatVis.visualize(
            robot_model=self.rmodel,
            robot_visual_model=self.vmodel,
            robot_collision_model=self.cmodel,
        )

    def update_croco_predictions(self, croco_xs, croco_us):
        self.croco_xs = croco_xs
        self.croco_us = croco_us

    def plot_traj(self):
        """Plot both trajectories of hpp and crocoddyl for the gripper pose."""
        pose_croco, pose_hpp = self.get_cartesian_trajectory()
        t = np.linspace(0, self.path_length, self.croco_xs.shape[0])
        axis_string = ["x", "y", "z"]
        for idx in range(3):
            plt.subplot(2, 2, idx + 1)
            plt.plot(t, pose_croco[idx])
            plt.plot(t, pose_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel("effector " + axis_string[idx] + " position")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_traj_configuration(self):
        """Plot both trajectories of hpp and crocoddyl in configuration space."""
        q_crocos = self.croco_xs[:, : self.nq]
        q_hpp = self.whole_x_plan[:, : self.nq]
        t = np.linspace(0, self.path_length, self.croco_xs.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, q_crocos[:, idx])
            plt.plot(t, q_hpp[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_traj_velocity(self):
        """Plot both velocities of hpp and crocoddyl."""
        v_crocos = self.croco_xs[:, self.nq :]
        v_hpp = self.whole_x_plan[:, self.nq :]
        t = np.linspace(0, self.path_length, self.croco_xs.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, v_crocos[:, idx])
            plt.plot(t, v_hpp[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"velocity q{idx}")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_integrated_configuration(self):
        """Plot both trajectories of hpp and crocoddyl in configuration space by integrating velocities."""
        v_crocos = self.croco_xs[:, self.nq :]
        v_hpp = self.whole_x_plan[:, self.nq :]
        q_crocos = [[] for _ in range(self.nq)]
        q_hpps = [[] for _ in range(self.nq)]

        # add initial configuration
        x0_croco = self.croco_xs[0]
        x0_hpp = self.whole_x_plan[0]
        for idx in range(self.nq):
            q_crocos[idx].append(x0_croco[idx])
            q_hpps[idx].append(x0_hpp[idx])

        # compute next configurations by integrating velocities
        for idx in range(self.croco_xs.shape[0]):
            for joint_idx in range(self.nq):
                q_croco = q_crocos[joint_idx][-1] + v_crocos[joint_idx][idx] * self.DT
                q_crocos[joint_idx].append(q_croco)
                q_hpp = q_hpps[joint_idx][-1] + v_hpp[joint_idx][idx] * self.DT
                q_hpps[joint_idx].append(q_hpp)

        t = np.linspace(0, self.path_length, self.croco_xs.shape[0] + 1)
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpps[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} integrated")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_control(self):
        """Plot control for each joint."""
        t = np.linspace(0, self.path_length, self.croco_us.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, self.croco_us[:, idx])
            plt.plot(t, self.whole_u_plan[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"tau{idx}")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def display_path_gepetto_gui(self):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        if self.rmodel.existJointName("panda_joint7") or self.rmodel.existJointName(
            "panda2_joint7"
        ):
            environnement_pose = [0, 0]  # value of gripper joints
        elif self.rmodel.existJointName("wrist_3_joint"):
            environnement_pose = [-0.2, 0, 0.02, 0, 0, 0, 1]  # pose of the ball
        for x in self.croco_xs:
            self.viewer(list(x)[: self.nq] + environnement_pose)
            time.sleep(self.DT)

    def display_path_meshcat(self):
        """Display in Meshcat the trajectory found with crocoddyl."""
        for x in self.croco_xs:
            self.vis[0].display(x[: self.nq])
            time.sleep(self.DT)

    def print_final_placement(self):
        """Print final gripper position for both hpp and crocoddyl trajectories."""
        q_final_hpp = self.whole_x_plan[-1][: self.nq]
        hpp_placement = get_ee_pose_from_configuration(
            self.rmodel, self._rdata, self._last_joint_frame_id, q_final_hpp
        )
        print("Last node placement ")
        print(
            "hpp rot ",
            pin.log(hpp_placement.rotation),
            " translation ",
            hpp_placement.translation,
        )
        q_final_croco = self.croco_xs[-1][: self.nq]
        croco_placement = get_ee_pose_from_configuration(
            self.rmodel, self._rdata, self._last_joint_frame_id, q_final_croco
        )
        print(
            "croco rot ",
            pin.log(croco_placement.rotation),
            " translation ",
            croco_placement.translation,
        )

    def get_trajectory_difference(self, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco = self.croco_xs[:, : self.nq]
            traj_hpp = self.prob.x_plan[:, : self.nq]  ### TODO for Théo
        else:
            traj_croco, traj_hpp = self.get_cartesian_trajectory()
        diffs = []
        for idx in range(len(traj_croco)):
            array_diff = np.abs(np.array(traj_croco[idx]) - np.array(traj_hpp[idx]))
            diffs.append(np.sum(array_diff))
        return sum(diffs)

    def get_cartesian_trajectory(self):
        """Return the vector of gripper pose for both trajectories found by hpp and crocoddyl."""
        pose_croco = [[] for _ in range(3)]
        pose_hpp = [[] for _ in range(3)]
        for idx in range(self.croco_xs.shape[0]):
            q = self.croco_xs[idx, : self.nq]
            pose = get_ee_pose_from_configuration(
                self.rmodel, self._rdata, self._last_joint_frame_id, q
            ).translation
            for idx in range(3):
                pose_croco[idx].append(pose[idx])
        for idx in range(self.whole_x_plan.shape[0]):
            q = self.whole_x_plan[idx, : self.nq]
            pin.framesForwardKinematics(self.rmodel, self._rdata, q)
            pose = get_ee_pose_from_configuration(
                self.rmodel, self._rdata, self._last_joint_frame_id, q
            ).translation
            for idx in range(3):
                pose_hpp[idx].append(pose[idx])
        return pose_croco, pose_hpp

    def plot_xs_us(self, solver):
        xs = np.array(solver.xs)
        us = np.array(solver.us)
        dt = solver.problem.runningModels[0].dt
        poses = np.zeros([len(xs), 3])
        for idx in range(xs.shape[0]):
            q_idx = xs[idx, : self.nq]
            pose = get_ee_pose_from_configuration(
                self.rmodel, self._rdata, self._last_joint_frame_id, q_idx
            ).translation
            poses[idx, :] = pose
        t_xs = np.linspace(0, (len(xs) - 1), int(1 / dt))
        # for idx in range(3):
        #    plt.subplot(3, 1, idx + 1)
        #    plt.plot(t_xs, poses[:, idx])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t_xs, xs[:, idx], label="q" + idx)
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t_xs[:-1], us[:, idx], label="u" + idx)
        plt.show()
