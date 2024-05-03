from problem import *
import matplotlib.pyplot as plt


class CrocoHppConnection:
    def __init__(self, ps, robot_name, vf, ball_init_pose):
        self.ball_init_pose = ball_init_pose
        if vf is not None:
            self.v = vf.createViewer()
        self.prob = Problem(ps, robot_name)
        self.robot = example_robot_data.load(robot_name)
        self.nq = self.robot.nq
        self.DT = self.prob.DT
        self.croco_xs = None
        self.hpp_paths = None

    def plot_traj(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl for the gripper pose."""
        pose_croco, pose_hpp = self.get_cartesian_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        axis_string = ["x", "y", "z"]
        for idx in range(3):
            plt.subplot(2, 2, idx + 1)
            plt.plot(t, pose_croco[idx])
            plt.plot(t, pose_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel("effector " + axis_string[idx] + " position")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_traj_configuration(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl in configuration space."""
        q_crocos, q_hpp = self.get_configuration_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_traj_velocity(self, terminal_idx):
        """Plot both velocities of hpp and crocoddyl."""
        v_crocos, v_hpp = self.get_velocity_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs))
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, v_crocos[idx])
            plt.plot(t, v_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"velocity q{idx}")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_integrated_configuration(self, terminal_idx):
        v_crocos, v_hpp = self.get_velocity_trajectory(terminal_idx)
        q_crocos = [[] for _ in range(self.nq)]
        q_hpps = [[] for _ in range(self.nq)]
        x0_croco = self.croco_xs[0]
        x0_hpp = self.hpp_paths[0].x_plan[0]
        for idx in range(self.nq):
            q_crocos[idx].append(x0_croco[idx])
            q_hpps[idx].append(x0_hpp[idx])

        for idx in range(len(self.croco_xs)):
            for joint_idx in range(self.nq):
                q_croco = q_crocos[joint_idx][-1] + v_crocos[joint_idx][idx] * self.DT
                q_crocos[joint_idx].append(q_croco)
                q_hpp = q_hpps[joint_idx][-1] + v_hpp[joint_idx][idx] * self.DT
                q_hpps[joint_idx].append(q_hpp)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.croco_xs) + 1)
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpps[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position integrated")
            plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def plot_control(self, terminal_idx):
        """Plot control for each joint."""
        us = self.get_configuration_control()
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, len(self.prob.solver.us))
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, us[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} control")
            # plt.legend(["crocoddyl", "hpp warm start"], loc="best")
        plt.show()

    def display_path(self, terminal_idx):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        path_time = self.get_path_length(terminal_idx)
        DT = path_time / len(self.croco_xs)
        for x in self.croco_xs:
            self.v(list(x)[: self.nq] + self.ball_init_pose)
            time.sleep(DT)

    def print_final_placement(self, terminal_idx):
        """Print final gripper position for both hpp and crocoddyl trajectories."""
        q_final_hpp = self.hpp_paths[terminal_idx].x_plan[-1][: self.nq]
        hpp_placement = self.robot.placement(q_final_hpp, self.nq)
        print("Last node placement ")
        print(
            "hpp rot ",
            pin.log(hpp_placement.rotation),
            " translation ",
            hpp_placement.translation,
        )
        q_final_croco = self.croco_xs[-1][: self.nq]
        croco_placement = self.robot.placement(q_final_croco, self.nq)
        print(
            "croco rot ",
            pin.log(croco_placement.rotation),
            " translation ",
            croco_placement.translation,
        )

    def get_path_length(self, terminal_idx):
        length = 0
        for idx in range(terminal_idx):
            length += self.prob.hpp_paths[idx].path.length()
        return length

    def get_trajectory_difference(self, terminal_idx, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco, traj_hpp = self.get_configuration_trajectory(terminal_idx)
        else:
            traj_croco, traj_hpp = self.get_cartesian_trajectory(terminal_idx)
        diffs = []
        for idx in range(len(traj_croco)):
            array_diff = np.abs(np.array(traj_croco[idx]) - np.array(traj_hpp[idx]))
            diffs.append(np.sum(array_diff))
        return sum(diffs)

    def get_cartesian_trajectory(self, terminal_idx):
        """Return the vector of gripper pose for both trajectories found by hpp and crocoddyl."""
        pose_croco = [[] for _ in range(3)]
        pose_hpp = [[] for _ in range(3)]
        for x in self.croco_xs:
            q = x[: self.nq]
            pose = self.robot.placement(q, self.nq).translation
            for idx in range(3):
                pose_croco[idx].append(pose[idx])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                q = x[: self.nq]
                pose = self.robot.placement(q, self.nq).translation
                for idx in range(3):
                    pose_hpp[idx].append(pose[idx])
        return pose_croco, pose_hpp

    def get_configuration_trajectory(self, terminal_idx):
        """Return the vector of configuration for both trajectories found by hpp and crocoddyl."""
        q_crocos = [[] for _ in range(self.nq)]
        q_hpp = [[] for _ in range(self.nq)]
        for x in self.croco_xs:
            for idx in range(self.nq):
                q_crocos[idx].append(x[idx])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                for idx in range(self.nq):
                    q_hpp[idx].append(x[idx])
        return q_crocos, q_hpp

    def get_velocity_trajectory(self, terminal_idx):
        """Return the vector of velocity for both trajectories found by hpp and crocoddyl."""
        v_crocos = [[] for _ in range(self.nq)]
        v_hpp = [[] for _ in range(self.nq)]
        for x in self.croco_xs:
            for idx in range(self.nq):
                v_crocos[idx].append(x[idx + self.nq])
        for path_idx in range(terminal_idx + 1):
            for x in self.hpp_paths[path_idx].x_plan:
                for idx in range(self.nq):
                    v_hpp[idx].append(x[idx + self.nq])
        return v_crocos, v_hpp

    def get_configuration_control(self):
        """Return the vector of configuration for both trajectories found by hpp and crocoddyl."""
        us = [[] for _ in range(self.nq)]
        for u in self.prob.solver.us:
            for idx in range(self.nq):
                us[idx].append(u[idx])
        return us

    def search_best_costs(self, terminal_idx, use_mim=False, configuration_traj=False):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        self.best_combination = None
        self.best_diff = 1e6
        self.max_control = 1e6
        self.best_solver = None
        if use_mim:
            for x_exponent in range(0, 8, 2):
                for u_exponent in range(-32, -26, 2):
                    start = time.time()
                    self.try_new_costs(
                        0,
                        x_exponent,
                        u_exponent,
                        terminal_idx,
                        use_mim,
                        configuration_traj,
                    )
                    end = time.time()
                    print("iteration duration ", end - start)
        else:
            for grip_exponent in range(34, 46, 2):
                print("grip expo ", grip_exponent)
                for x_exponent in range(-20, 0, 5):
                    for u_exponent in range(-35, -15, 5):
                        start = time.time()
                        self.try_new_costs(
                            grip_exponent,
                            x_exponent,
                            u_exponent,
                            terminal_idx,
                            use_mim=use_mim,
                            configuration_traj=configuration_traj,
                        )
                        end = time.time()
                        print("iteration duration ", end - start)
            self.max_control = np.max(self.prob.solver.us)
            best_combination = self.best_combination
            """
            for vel_exponent in range(-40, -5, 5):
                for xlim_exponent in range(-20, -5, 5):
                    self.try_new_costs(
                        best_combination[0],
                        best_combination[1],
                        best_combination[2],
                        terminal_idx,
                        vel_exponent,
                        xlim_exponent,
                        use_mim,
                        configuration_traj,
                    )"""

        self.prob.solver = self.best_solver
        self.croco_xs = self.prob.solver.xs
        print("best diff ", self.best_diff)
        print("best combination ", self.best_combination)
        print("max u ", np.max(self.prob.solver.us))

    def try_new_costs(
        self,
        grip_exponent,
        x_exponent,
        u_exponent,
        terminal_idx,
        vel_exponent=0,
        xlim_exponent=0,
        use_mim=False,
        configuration_traj=False,
    ):
        """Set problem, run solver and check if we found a better solution."""
        self.set_problem_run_solver(
            10 ** (grip_exponent / 10),
            10 ** (x_exponent / 10),
            10 ** (u_exponent / 10),
            10 ** (vel_exponent / 10),
            10 ** (xlim_exponent / 10),
            terminal_idx,
            use_mim,
        )
        diff = self.get_trajectory_difference(terminal_idx, configuration_traj)
        if diff < self.best_diff:
            self.best_combination = [
                grip_exponent,
                x_exponent,
                u_exponent,
                vel_exponent,
                xlim_exponent,
            ]
            self.best_diff = diff
            self.best_solver = self.prob.solver

    def set_problem_run_solver(
        self,
        grip_cost,
        x_cost,
        u_cost,
        vel_cost,
        xlim_cost,
        terminal_idx,
        use_mim=False,
    ):
        """Set OCP problem with new costs then run solver."""
        self.prob.set_costs(grip_cost, x_cost, u_cost, vel_cost, xlim_cost)
        self.prob.set_models([terminal_idx], use_mim=use_mim)
        self.prob.run_solver(0, terminal_idx, use_mim=use_mim)
        self.croco_xs = self.prob.solver.xs
        self.hpp_paths = self.prob.hpp_paths
