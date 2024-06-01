import time
import numpy as np
import matplotlib.pyplot as plt

import pybullet
from mim_robots.pybullet.env import BulletEnvWithGround
import pinocchio as pin
import mpc_utils
import pin_utils
from ocp import OCPPandaReachingColWithMultipleCol
from wrapper_panda import PandaRobot
from scenes import Scene

np.set_printoptions(precision=4, linewidth=180)


class MPC:
    def __init__(
        self,
        robot_simulator: PandaRobot,
        OCP: OCPPandaReachingColWithMultipleCol,
        max_iter: int,
        env: BulletEnvWithGround,
        TARGET_POSE_1: pin.SE3,
        TARGET_POSE_2: pin.SE3,
    ):
        """Create the MPC class used to solve the OCP problem.

        Args:
            robot_simulator (PandaRobot): Wrapper of pinocchio and pybullet robot.
            OCP (OCPPandaReachingColWithMultipleCol): OCP solved in the MPC.
            max_iter (int): Number max of iterations of the solver.
            env (BulletEnvWithGround): Pybullet environement.
            TARGET_POSE_1 (pin.SE3): First target of the robot.
            TARGET_POSE_2 (pin.SE3): Second target of the robot.
        """

        # Robot wrapper
        self._robot_simulator = robot_simulator

        # Environement of the robot
        self._env = env
        self._scene = self._robot_simulator.scene

        # Setting up the OCP to be solved
        self._OCP = OCP
        self._sqp = self._OCP()

        # OCP params needed
        self._T = self._OCP._T  # Number of nodes
        self._dt = self._OCP._dt  # Time between each node
        self._x0 = self._OCP._x0  # Initial state of the robot
        self._xs_init = [
            self._x0 for i in range(self._T + 1)
        ]  # Initial trajectory of the robot
        self._us_init = self._sqp.problem.quasiStatic(
            self._xs_init[:-1]
        )  # Initial command of the robot
        self._max_iter = max_iter  # Number max of iterations of the solver
        self._TARGET_POSE_1 = TARGET_POSE_1  # First target of the robot
        self._TARGET_POSE_2 = TARGET_POSE_2  # Second target of the robot

        # Filling the OCP dictionnary used to store all the results
        self._ocp_params = {}
        self._ocp_params["N_h"] = self._T
        self._ocp_params["dt"] = self._dt
        self._ocp_params["maxiter"] = self._max_iter
        self._ocp_params["pin_model"] = robot_simulator.pin_robot.model
        self._ocp_params["armature"] = self._OCP._runningModel.differential.armature
        self._ocp_params["id_endeff"] = robot_simulator.pin_robot.model.getFrameId(
            "panda2_leftfinger"
        )
        self._ocp_params["active_costs"] = self._sqp.problem.runningModels[
            0
        ].differential.costs.active.tolist()

        # Filling the simu parameters dictionnary
        self._sim_params = {}
        self._sim_params["sim_freq"] = int(1.0 / env.dt)
        self._sim_params["mpc_freq"] = 1000
        self._sim_params["T_sim"] = 0.01
        self._log_rate = 100

        # Initialize simulation data
        self._sim_data = mpc_utils.init_sim_data(
            self._sim_params, self._ocp_params, self._x0
        )

        # Display target in pybullet
        mpc_utils.display_ball(
            self._TARGET_POSE_1.translation, RADIUS=0.05, COLOR=[1.0, 0.0, 0.0, 0.6]
        )
        mpc_utils.display_ball(
            self._TARGET_POSE_2.translation, RADIUS=0.5e-1, COLOR=[1.0, 0.0, 0.0, 0.6]
        )

        # Boolean describing whether the MPC has already been solved or not
        self._SOLVE = False

    def solve(self):
        """Solve the MPC problem and display it in a pybullet GUI."""

        time_calc = []
        u_list = []
        # Simulate
        mpc_cycle = 0
        TARGET_POSE = self._TARGET_POSE_1
        for i in range(self._sim_data["N_sim"]):
            if i % 500 == 0 and i != 0:
                ### Changing from target pose 1 to target pose 2 or inversely
                if TARGET_POSE == self._TARGET_POSE_1:
                    TARGET_POSE = self._TARGET_POSE_2
                else:
                    TARGET_POSE = self._TARGET_POSE_1

                for k in range(self._T):
                    self._sqp.problem.runningModels[k].differential.costs.costs[
                        "gripperPoseRM"
                    ].cost.residual.reference = TARGET_POSE.translation
                self._sqp.problem.terminalModel.differential.costs.costs[
                    "gripperPose"
                ].cost.residual.reference = TARGET_POSE.translation

            if i % self._log_rate == 0:
                print(
                    "\n SIMU step " + str(i) + "/" + str(self._sim_data["N_sim"]) + "\n"
                )

            # Solve OCP if we are in a planning cycle (MPC/planning frequency)
            if (
                i % int(self._sim_params["sim_freq"] / self._sim_params["mpc_freq"])
                == 0
            ):
                # Set x0 to measured state
                self._sqp.problem.x0 = self._sim_data["state_mea_SIM_RATE"][i, :]
                # Warm start using previous solution
                xs_init = [*list(self._sqp.xs[1:]), self._sqp.xs[-1]]
                xs_init[0] = self._sim_data["state_mea_SIM_RATE"][i, :]
                us_init = [*list(self._sqp.us[1:]), self._sqp.us[-1]]

                # Solve OCP & record MPC predictions
                start = time.process_time()
                self._sqp.solve(xs_init, us_init, maxiter=self._ocp_params["maxiter"])
                t_solve = time.process_time() - start
                time_calc.append(t_solve)
                self._sim_data["state_pred"][mpc_cycle, :, :] = np.array(self._sqp.xs)
                self._sim_data["ctrl_pred"][mpc_cycle, :, :] = np.array(self._sqp.us)
                # Extract relevant predictions for interpolations
                x_curr = self._sim_data["state_pred"][
                    mpc_cycle, 0, :
                ]  # x0* = measured state    (q^,  v^ )
                x_pred = self._sim_data["state_pred"][
                    mpc_cycle, 1, :
                ]  # x1* = predicted state   (q1*, v1*)
                u_curr = self._sim_data["ctrl_pred"][
                    mpc_cycle, 0, :
                ]  # u0* = optimal control   (tau0*)
                # Record costs references
                q = self._sim_data["state_pred"][mpc_cycle, 0, : self._sim_data["nq"]]
                self._sim_data["ctrl_ref"][mpc_cycle, :] = pin_utils.get_u_grav(
                    q,
                    self._sqp.problem.runningModels[0].differential.pinocchio,
                    self._ocp_params["armature"],
                )
                self._sim_data["state_ref"][mpc_cycle, :] = (
                    self._sqp.problem.runningModels[0]
                    .differential.costs.costs["stateReg"]
                    .cost.residual.reference
                )
                self._sim_data["lin_pos_ee_ref"][mpc_cycle, :] = (
                    self._sqp.problem.runningModels[0]
                    .differential.costs.costs["gripperPoseRM"]
                    .cost.residual.reference
                )

                # Select reference control and state for the current MPC cycle
                x_ref_MPC_RATE = x_curr + self._sim_data["ocp_to_mpc_ratio"] * (
                    x_pred - x_curr
                )
                u_ref_MPC_RATE = u_curr
                if mpc_cycle == 0:
                    self._sim_data["state_des_MPC_RATE"][mpc_cycle, :] = x_curr
                self._sim_data["ctrl_des_MPC_RATE"][mpc_cycle, :] = u_ref_MPC_RATE
                self._sim_data["state_des_MPC_RATE"][mpc_cycle + 1, :] = x_ref_MPC_RATE

                # Increment planning counter
                mpc_cycle += 1

                # Select reference control and state for the current SIMU cycle
                x_ref_SIM_RATE = x_curr + self._sim_data["ocp_to_mpc_ratio"] * (
                    x_pred - x_curr
                )
                u_ref_SIM_RATE = u_curr

                # First prediction = measurement = initialization of MPC
                if i == 0:
                    self._sim_data["state_des_SIM_RATE"][i, :] = x_curr
                self._sim_data["ctrl_des_SIM_RATE"][i, :] = u_ref_SIM_RATE
                self._sim_data["state_des_SIM_RATE"][i + 1, :] = x_ref_SIM_RATE

                # Send torque to simulator & step simulator
                self._robot_simulator.send_joint_command(u_ref_SIM_RATE)
                self._env.step()
                # Measure new state from simulator
                q_mea_SIM_RATE, v_mea_SIM_RATE = self._robot_simulator.get_state()
                # Update pinocchio model
                self._robot_simulator.forward_robot(q_mea_SIM_RATE, v_mea_SIM_RATE)
                # Record data
                x_mea_SIM_RATE = np.concatenate([q_mea_SIM_RATE, v_mea_SIM_RATE]).T
                self._sim_data["state_mea_SIM_RATE"][i + 1, :] = x_mea_SIM_RATE
                u_list.append(u_curr.tolist())

        self._SOLVE = True

    def plot_collision_distances(self):
        """Plot the distances between obstacles and shapes of the robot throughout the whole trajectory.

        Raises:
            NotSolvedError: Call the solve method before the plot one.
        """
        if not self._SOLVE:
            raise NotSolvedError()

        self._shapes_in_collision_with_obstacle = (
            self._scene.get_shapes_avoiding_collision()
        )
        self._obstacles = self._scene._obstacles_name

        self._distances = {}

        # Creating the dictionnary regrouping the distances
        for obstacle in self._obstacles:
            self._distances[obstacle] = {}
            for shapes in self._shapes_in_collision_with_obstacle:
                self._distances[obstacle][shapes] = []

        # Going through all the trajectory
        for q in self._sim_data["state_mea_SIM_RATE"]:
            # Going through the obstacles
            for obstacle, shape in self._distances.items():
                for shape_name, distance_between_shape_and_obstacle in shape.items():
                    id_shape = robot_simulator.pin_robot.collision_model.getGeometryId(
                        shape_name
                    )
                    id_obstacle = (
                        robot_simulator.pin_robot.collision_model.getGeometryId(
                            obstacle
                        )
                    )
                    dist = pin_utils.compute_distance_between_shapes(
                        robot_simulator.pin_robot.model,
                        robot_simulator.pin_robot.collision_model,
                        id_shape,
                        id_obstacle,
                        q[:7],
                    )
                    distance_between_shape_and_obstacle.append(dist)

        # Doing the plot blakc magic
        ncols = 2
        nrows = (len(self._obstacles) + 1) // 2 if len(self._obstacles) > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))

        # Flatten axes array for consistent indexing
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1 or ncols == 1:
            axes = (
                np.expand_dims(axes, axis=0)
                if nrows == 1
                else np.expand_dims(axes, axis=1)
            )
        else:
            axes = axes.reshape((nrows, ncols))

        # Flatten the 2D axes array for easy iteration
        flat_axes = axes.flatten()

        for ax, (obstacle, shapes) in zip(flat_axes, self._distances.items()):
            for shape, values in shapes.items():
                (handle,) = ax.plot(values, label=shape)
            ax.plot(np.zeros(len(values)), "-", label="Collision threshold")
            ax.set_ylabel("Distance from the obstacle")
            ax.set_xlabel("Timestep of the simulation")
            ax.set_title(f"Collisions with {obstacle}")
    
        # Create a single legend outside the plots
        if nrows >1:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(-0.15, -0.3),
                fancybox=True,
                shadow=True,
                ncol=8,
                prop={'size': 6}
            )
        else:
            ax.legend()

        # Hide any unused subplots
        for i in range(len(self._distances), len(flat_axes)):
            fig.delaxes(flat_axes[i])

        # Adjust the layout to make room for legend and title
        plt.suptitle("Distance between collision shapes & obstacles")
        fig.subplots_adjust(hspace=0.5)
        plt.show()


class NotSolvedError(Exception):
    """Exception raised when plot is called before solve for the MPC class."""

    def __init__(self, message="Solve method must be called before plot."):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":

    # # # # # # # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL and SIMU ENV ###
    # # # # # # # # # # # # # # # # # # #

    # Name of the scene
    name_scene = "box"

    # Pose of the obstacle

    # Creation of the scene
    scene = Scene(name_scene=name_scene)

    # Simulation environment
    env = BulletEnvWithGround(server=pybullet.GUI, dt=1e-3)
    # Robot simulator
    robot_simulator = PandaRobot(
        capsule=True, auto_col=True, pos_obs=scene.obstacle_pose, name_scene=name_scene
    )
    TARGET_POSE1, TARGET_POSE2, q0 = (
        robot_simulator.TARGET_POSE1,
        robot_simulator.TARGET_POSE2,
        robot_simulator.q0,
    )
    # Creating the scene

    env.add_robot(robot_simulator)

    # Extract robot model
    nq = robot_simulator.pin_robot.model.nq
    nv = robot_simulator.pin_robot.model.nv
    nu = nq
    nx = nq + nv
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])
    # Add robot to simulation and initialize
    robot_simulator.reset_state(q0, v0)
    robot_simulator.forward_robot(q0, v0)
    print("[PyBullet] Created robot (id = " + str(robot_simulator.robotId) + ")")

    dt = 2e-2
    T = 2

    max_iter = 4  # Maximum iterations of the solver
    max_qp_iters = 25  # Maximum iterations for solving each qp solved in one iteration of the solver

    WEIGHT_GRIPPER_POSE = 1e2
    WEIGHT_GRIPPER_POSE_TERM = 1e2
    WEIGHT_xREG = 1e-2
    WEIGHT_xREG_TERM = 1e-2
    WEIGHT_uREG = 1e-4
    max_qp_iters = 25
    callbacks = False
    safety_threshhold = 7e-2

    ### CREATING THE PROBLEM WITH OBSTACLE

    print("Solving the problem with collision")
    problem = OCPPandaReachingColWithMultipleCol(
        robot_simulator.pin_robot.model,
        robot_simulator.pin_robot.collision_model,
        TARGET_POSE1,
        T,
        dt,
        x0,
        WEIGHT_xREG=WEIGHT_xREG,
        WEIGHT_uREG=WEIGHT_uREG,
        WEIGHT_GRIPPER_POSE=WEIGHT_GRIPPER_POSE,
        MAX_QP_ITERS=max_qp_iters,
        SAFETY_THRESHOLD=safety_threshhold,
    )

    mpc = MPC(
        robot_simulator,
        OCP=problem,
        max_iter=max_iter,
        env=env,
        TARGET_POSE_1=TARGET_POSE1,
        TARGET_POSE_2=TARGET_POSE2,
    )
    mpc.solve()
    mpc.plot_collision_distances()
