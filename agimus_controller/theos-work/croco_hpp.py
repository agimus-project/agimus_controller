from .problem import *
import matplotlib.pyplot as plt


class CrocoHppConnection:
    def __init__(self, ps, robot_name, vf, ball_init_pose):
        self.ball_init_pose = ball_init_pose
        if vf is not None:
            self.v = vf.createViewer()
        self.prob = Problem(ps, robot_name)
        self.robot = self.prob.robot
        self.nq = self.robot.nq
        self.DT = self.prob.DT
        self.croco_xs = None
        self.croco_us = None
        self.results = {}
        self.results["xs"] = []
        self.results["us"] = []
        self.results["max_us"] = []
        self.results["max_increase_us"] = []
        self.results["combination"] = []
        self.hpp_paths = self.prob.hpp_paths

    def plot_traj(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl for the gripper pose."""
        pose_croco, pose_hpp = self.get_cartesian_trajectory(terminal_idx)
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, self.croco_xs.shape[0])
        axis_string = ["x", "y", "z"]
        for idx in range(3):
            plt.subplot(2, 2, idx + 1)
            plt.plot(t, pose_croco[idx])
            plt.plot(t, pose_hpp[idx])
            plt.xlabel("time (s)")
            plt.ylabel("effector " + axis_string[idx] + " position")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_traj_configuration(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl in configuration space."""
        q_crocos = self.croco_xs[:, : self.nq]
        q_hpp = self.prob.x_plan[:, : self.nq]
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, self.croco_xs.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, q_crocos[:, idx])
            plt.plot(t, q_hpp[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} position")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_traj_velocity(self, terminal_idx):
        """Plot both velocities of hpp and crocoddyl."""
        v_crocos = self.croco_xs[:, self.nq :]
        v_hpp = self.prob.x_plan[:, self.nq :]
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, self.croco_xs.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.robot.nq, 1, idx + 1)
            plt.plot(t, v_crocos[:, idx])
            plt.plot(t, v_hpp[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"velocity q{idx}")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_integrated_configuration(self, terminal_idx):
        """Plot both trajectories of hpp and crocoddyl in configuration space by integrating velocities."""
        v_crocos = self.croco_xs[:, self.nq :]
        v_hpp = self.prob.x_plan[:, self.nq :]
        q_crocos = [[] for _ in range(self.nq)]
        q_hpps = [[] for _ in range(self.nq)]

        # add initial configuration
        x0_croco = self.croco_xs[0]
        x0_hpp = self.hpp_paths[0].x_plan[0]
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

        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, self.croco_xs.shape[0] + 1)
        for idx in range(self.nq):
            plt.subplot(3, 2, idx + 1)
            plt.plot(t, q_crocos[idx])
            plt.plot(t, q_hpps[idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} integrated")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def plot_control(self, terminal_idx):
        """Plot control for each joint."""
        path_time = self.get_path_length(terminal_idx)
        t = np.linspace(0, path_time, self.croco_us.shape[0])
        for idx in range(self.nq):
            plt.subplot(self.nq, 1, idx + 1)
            plt.plot(t, self.croco_us[:, idx])
            plt.plot(t, self.prob.u_ref[:, idx])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{idx} control")
            plt.legend(["crocoddyl", "hpp"], loc="best")
        plt.show()

    def display_path(self, terminal_idx):
        """Display in Gepetto Viewer the trajectory found with crocoddyl."""
        path_time = self.get_path_length(terminal_idx)
        DT = path_time / self.croco_xs.shape[0]
        for x in self.croco_xs:
            self.v(list(x)[: self.nq] + self.ball_init_pose)  # + self.ball_init_pose
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
        for idx in range(terminal_idx + 1):
            length += self.prob.hpp_paths[idx].path.length()
        return length

    def get_trajectory_difference(self, terminal_idx, configuration_traj=True):
        """Compute at each node the absolute difference in position either in cartesian or configuration space and sum it."""
        if configuration_traj:
            traj_croco = self.croco_xs[:, : self.nq]
            traj_hpp = self.prob.x_plan[:, : self.nq]
        else:
            traj_croco, traj_hpp = self.get_cartesian_trajectory(terminal_idx)
        diffs = []
        for idx in range(len(traj_croco)):
            array_diff = np.abs(np.array(traj_croco[idx]) - np.array(traj_hpp[idx]))
            diffs.append(np.sum(array_diff))
        return sum(diffs)

    def max_increase_us(self):
        """Return control max increase"""
        increases = np.zeros([self.croco_us.shape[0] - 1, self.robot.nq])
        for joint_idx in range(self.robot.nq):
            for idx in range(self.croco_us.shape[0] - 1):
                increases[idx, joint_idx] = (
                    self.croco_us[idx + 1, joint_idx] - self.croco_us[idx, joint_idx]
                )
        return np.max(np.abs(increases)), np.unravel_index(
            np.argmax(np.abs(increases), axis=None), increases.shape
        )

    def get_cartesian_trajectory(self, terminal_idx):
        """Return the vector of gripper pose for both trajectories found by hpp and crocoddyl."""
        pose_croco = [[] for _ in range(3)]
        pose_hpp = [[] for _ in range(3)]
        for idx in range(self.croco_xs.shape[0]):
            q = self.croco_xs[idx, : self.nq]
            pose = self.robot.placement(q, self.nq).translation
            for idx in range(3):
                pose_croco[idx].append(pose[idx])
        for path_idx in range(terminal_idx + 1):
            for idx in range(self.hpp_paths[path_idx].x_plan.shape[0]):
                q = self.hpp_paths[path_idx].x_plan[idx, : self.nq]
                pose = self.robot.placement(q, self.nq).translation
                for idx in range(3):
                    pose_hpp[idx].append(pose[idx])
        return pose_croco, pose_hpp

    def search_best_costs(
        self, terminal_idx, use_mim=False, configuration_traj=False, is_mpc=False
    ):
        """Search costs that minimize the gap between hpp and crocoddyl trajectories."""
        self.best_combination = None
        self.best_croco_xs = None
        self.best_croco_us = None
        self.best_diff = 1e6
        self.prob.use_mim = use_mim
        if use_mim:
            for x_exponent in range(0, 8, 2):
                for u_exponent in range(-32, -26, 2):
                    start = time.time()
                    _, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                        0, x_exponent, u_exponent, 0, 0
                    )
                    self.try_new_costs(
                        grip_cost,
                        x_cost,
                        u_cost,
                        terminal_idx,
                        0,
                        0,
                        configuration_traj=configuration_traj,
                        vel_cost=0,
                        xlim_cost=0,
                        is_mpc=is_mpc,
                    )
                    end = time.time()
                    print("iteration duration ", end - start)
        else:
            for grip_exponent in range(25, 50, 5):
                print("grip expo ", grip_exponent)
                for x_exponent in range(0, 15, 5):
                    for u_exponent in range(-35, -25, 5):
                        # for vel_exponent in range(-6, 14, 4):
                        grip_cost, x_cost, u_cost, _, _ = self.get_cost_from_exponent(
                            grip_exponent, x_exponent, u_exponent, 0, 0
                        )
                        self.prob.set_costs(grip_cost, x_cost, u_cost, 0, 0)
                        start = time.time()
                        self.try_new_costs(
                            grip_cost,
                            x_cost,
                            u_cost,
                            terminal_idx,
                            configuration_traj=configuration_traj,
                            vel_cost=0,
                            xlim_cost=0,
                            is_mpc=is_mpc,
                        )
                        end = time.time()
                        print("iteration duration ", end - start)

        self.croco_xs = self.best_croco_xs
        self.croco_us = self.best_croco_us
        print("best diff ", self.best_diff)
        print("best combination ", self.best_combination)
        print("max torque ", np.max(np.abs(self.croco_us)))

    def get_cost_from_exponent(
        self, grip_exponent, x_exponent, u_exponent, vel_exponent=0, xlim_exponent=0
    ):
        return (
            10 ** (grip_exponent / 10),
            10 ** (x_exponent / 10),
            10 ** (u_exponent / 10),
            10 ** (vel_exponent / 10),
            10 ** (xlim_exponent / 10),
        )

    def try_new_costs(
        self,
        grip_cost,
        x_cost,
        u_cost,
        terminal_idx,
        vel_cost=0,
        xlim_cost=0,
        configuration_traj=False,
        is_mpc=False,
    ):
        """Set problem, run solver, add result in dict and check if we found a better solution."""
        if is_mpc:
            print("doing mpc")
            self.do_mpc(terminal_idx, 100)
        else:
            print("doing ocp")
            self.set_problem_run_solver(
                terminal_idx,
            )
        max_us = np.max(np.abs(self.croco_us))
        max_increase_us, _ = self.max_increase_us()
        self.results["xs"].append(self.croco_xs)
        self.results["us"].append(self.croco_us)
        self.results["max_us"].append(max_us)
        self.results["max_increase_us"].append(max_increase_us)
        self.results["combination"].append(
            [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
        )
        diff = self.get_trajectory_difference(terminal_idx, configuration_traj)
        if diff < self.best_diff and max_us < 100 and max_increase_us < 50:
            self.best_combination = [grip_cost, x_cost, u_cost, vel_cost, xlim_cost]
            self.best_diff = diff
            self.best_croco_xs = self.croco_xs
            self.best_croco_us = self.croco_us

    def set_problem_run_solver(self, terminal_idx):
        """Set OCP problem with new costs then run solver."""

        self.prob.set_models([terminal_idx])
        self.prob.create_whole_problem()
        self.prob.set_xplan_and_uref(0, terminal_idx)
        self.prob.run_solver(
            self.prob.whole_problem, self.prob.x_plan, self.prob.u_ref, 10
        )
        self.croco_xs = np.array(self.prob.solver.xs)
        self.croco_us = np.array(self.prob.solver.us)

    def compute_next_step(self, x, problem):
        m = problem.runningModels[0]
        d = m.createData()
        m.calc(d, x, self.prob.solver.us[0])
        return d.xnext.copy()

    def do_mpc(self, path_terminal_idx, T, node_idx_breakpoint=None):
        self.prob.set_models([path_terminal_idx])
        self.prob.create_whole_problem()
        self.prob.set_xplan_and_uref(0, path_terminal_idx)

        problem = self.prob.create_problem(T)
        problem.x0 = self.hpp_paths[0].x_plan[0]
        self.prob.run_solver(
            problem, list(self.prob.x_plan[:T]), list(self.prob.u_ref[: T - 1]), 1000
        )
        next_node_idx = T
        xs = np.zeros([len(self.prob.whole_problem.runningModels) + 1, 2 * self.nq])
        xs[0, :] = problem.x0
        us = np.zeros([len(self.prob.whole_problem.runningModels), self.nq])
        us[0, :] = self.prob.solver.us[0]
        x = self.compute_next_step(problem.x0, problem)
        xs[1, :] = x
        breakpoint()
        for idx in range(1, len(self.prob.whole_problem.runningModels)):
            self.prob.reset_ocp(x, next_node_idx)
            xs_init = list(self.prob.solver.xs[1:]) + [self.prob.solver.xs[-1]]
            xs_init[0] = x
            us_init = list(self.prob.solver.us[1:]) + [self.prob.solver.us[-1]]

            self.prob.run_solver(problem, xs_init, us_init, 1)
            x = self.compute_next_step(x, problem)
            xs[idx + 1, :] = x
            us[idx, :] = self.prob.solver.us[0]
            next_node_idx += 1
            self.prob.solver.problem.copy()
            if idx == node_idx_breakpoint:
                breakpoint()
        self.croco_xs = xs
        self.croco_us = us
