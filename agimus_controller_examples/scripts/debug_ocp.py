from pathlib import Path
import example_robot_data as robex
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pinocchio as pin

from agimus_controller.ocp.ocp_croco_goal_reaching import OCPCrocoGoalReaching
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.trajectory import TrajectoryPoint
from agimus_controller.warm_start_reference import WarmStartReference

# Constants
SINUSOID_AMPLITUDE = 0.2
SINUSOID_FREQUENCY = 0.5 * np.pi
CONFIGURATION_WEIGHT_LIST = [np.sqrt(1), 1e-1]  # List of configuration weights to test
CONFIGURATION_WEIGHT_LIST = [1]
# VELOCITY_WEIGHT = 0.01
EFFORT_WEIGHT_LIST = [
    np.sqrt(1e-10),
    1e-6,
    1e-5,
    1e-4,
    1e-3,
]  # List of effort weights to test
EFFORT_WEIGHT_LIST = [0]
DT = 0.01
HORIZON_SIZE = 20
SOLVER_ITERS = 10
CALLBACKS = True


with open("slow_sim_weighted_trajectory_data.pkl", "rb") as pickle_file:
    weighted_trajectory_data = pickle.load(pickle_file)
    for i in range(len(weighted_trajectory_data)):
        weighted_trajectory_data[i].point.end_effector_poses["panda_joint7"] = (
            weighted_trajectory_data[i].point.end_effector_poses.pop("fer_joint7")
        )
        weighted_trajectory_data[i].weights.w_end_effector_poses["panda_joint7"] = (
            weighted_trajectory_data[i].weights.w_end_effector_poses.pop("fer_joint7")
        )


Q0 = weighted_trajectory_data[0].point.robot_configuration


def generate_sinusoide_variation(q, t):
    """
    Generates a sinusoidal variation in the robot's configuration.
    """
    for i in [2]:  # Apply sinusoidal variation to the 3rd joint
        q[i] = Q0[i] + SINUSOID_AMPLITUDE * np.sin(SINUSOID_FREQUENCY * t * DT)
    return q


# def generate_trajectory(robot_models, ocp_params, q0, ee_pose, configuration_weight, effort_weight):
#     """
#     Generates a reference trajectory and warm-start states/controls for the OCP.
#     """
#     state_warmstart = []
#     control_warmstart = []
#     trajectory_points = []
#     q_t = Q0.copy()
#     q_t_list = [q_t.copy()]

#     for i in range(1, ocp_params.horizon_size):
#         u_ref = np.zeros(robot_models.robot_model.nv)
#         q_t = generate_sinusoide_variation(q_t, i)
#         q_t_list.append(q_t.copy())
#         trajectory_points.append(
#             WeightedTrajectoryPoint(
#                 TrajectoryPoint(
#                     robot_configuration=q_t.copy(),
#                     robot_velocity=np.zeros(robot_models.robot_model.nv),
#                     robot_effort=u_ref,
#                     end_effector_poses={"panda_hand_tcp": ee_pose},
#                 ),
#                 TrajectoryPointWeights(
#                     w_robot_configuration=configuration_weight * np.ones(robot_models.robot_model.nq),
#                     w_robot_velocity=VELOCITY_WEIGHT * np.ones(robot_models.robot_model.nv),
#                     w_robot_effort=effort_weight * np.ones(robot_models.robot_model.nv),
#                     w_end_effector_poses={
#                         "panda_hand_tcp": (
#                             0 * np.ones(6)
#                             if i < ocp_params.horizon_size - 1
#                             else 0 * np.ones(6)
#                         )
#                     },
#                 ),
#             )
#         )
#         state_warmstart.append(np.concatenate((q_t, np.zeros(robot_models.robot_model.nv))))
#         if i != 0:
#             control_warmstart.append(u_ref)

#     return trajectory_points, state_warmstart, control_warmstart, q_t_list

# Main script
if __name__ == "__main__":
    robot = robex.load("panda")

    urdf_path = Path(robot.urdf)
    srdf_path = Path(robot.urdf.replace("urdf", "srdf"))
    urdf_meshes_dir = urdf_path.parent.parent.parent.parent.parent
    free_flyer = False
    locked_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    moving_joint_names = set(robot.model.names) - set(locked_joint_names + ["universe"])

    armature = np.full(robot.model.nq - len(locked_joint_names), 0.1)

    params = RobotModelParameters(
        q0=pin.neutral(robot.model),
        free_flyer=free_flyer,
        moving_joint_names=moving_joint_names,
        urdf=urdf_path,
        srdf=srdf_path,
        urdf_meshes_dir=urdf_meshes_dir,
        collision_as_capsule=True,
        self_collision=False,
        armature=armature,
    )

    robot_models = RobotModels(params)
    ws = WarmStartReference()
    ws.setup(robot_models._robot_model)
    rmodel = robot_models._robot_model
    rdata = rmodel.createData()

    # OCP parameters
    ocp_params = OCPParamsBaseCroco(
        dt=DT,
        horizon_size=HORIZON_SIZE,
        solver_iters=10000,
        callbacks=CALLBACKS,
        termination_tolerance=1e-9,
        eps_abs=1e-9,
        eps_rel=1e-9,
    )

    for i in range(len(weighted_trajectory_data)):
        weighted_trajectory_data[i].weights.w_robot_velocity = 0 * np.ones_like(
            weighted_trajectory_data[i].weights.w_robot_velocity
        )
        weighted_trajectory_data[i].weights.w_robot_acceleration = 0 * np.ones_like(
            weighted_trajectory_data[i].weights.w_robot_acceleration
        )
        # TODO: VP
        weighted_trajectory_data[i].weights.w_end_effector_poses["panda_joint7"] *= 0

    # Iterate over configuration weights
    for configuration_weight in CONFIGURATION_WEIGHT_LIST:
        for i in range(len(weighted_trajectory_data)):
            weighted_trajectory_data[i].weights.w_robot_configuration = (
                configuration_weight
                * np.ones_like(
                    weighted_trajectory_data[i].weights.w_robot_configuration
                )
            )
            # weighted_trajectory_data[i].weights.w_robot_configuration = np.array([0., 0., 1., 0., 0., 0., 0.])
        print(f"Solving OCP for configuration_weight = {configuration_weight}...")
        results = {}

        # Iterate over effort weights
        for effort_weight in EFFORT_WEIGHT_LIST:
            print(f"  Solving OCP for effort_weight = {effort_weight}...")
            for i in range(len(weighted_trajectory_data)):
                weighted_trajectory_data[i].weights.w_robot_effort = (
                    1e-3
                    * np.ones_like(weighted_trajectory_data[i].weights.w_robot_effort)
                )

            q_2_traj = []
            # current_state = weighted_trajectory_data[0].point
            for i in range(len(weighted_trajectory_data) - HORIZON_SIZE):
                # Solve OCP
                # x0 = np.concatenate((
                #     weighted_trajectory_data[i].point.robot_configuration,
                #     weighted_trajectory_data[i].point.robot_velocity))

                ocp = OCPCrocoGoalReaching(robot_models, ocp_params)
                reference_trajectory = weighted_trajectory_data[i : i + HORIZON_SIZE]
                ocp.set_reference_weighted_trajectory(reference_trajectory)
                reference_trajectory_points = [el.point for el in reference_trajectory]
                xo_ref = TrajectoryPoint(
                    robot_configuration=reference_trajectory_points[
                        0
                    ].robot_configuration.copy(),
                    robot_velocity=reference_trajectory_points[0].robot_velocity.copy(),
                    robot_acceleration=reference_trajectory_points[
                        0
                    ].robot_acceleration.copy(),
                )
                # xo_ref.robot_configuration[1] += np.random.random() * 0.1
                # xo_ref.robot_configuration[2] -=0.05
                # xo_ref.robot_configuration[3] +=0.05
                x0, x_init, u_init = ws.generate(
                    # reference_trajectory_points[0], reference_trajectory_points
                    xo_ref,
                    reference_trajectory_points,
                )

                import copy

                rr = copy.deepcopy(reference_trajectory)
                for jj in range(len(rr) - 1):
                    rr[jj].point.robot_effort = u_init[jj]
                ocp.set_reference_weighted_trajectory(rr)

                print([point.robot_effort[2] for point in reference_trajectory_points])
                print([u[2] for u in u_init])
                # print([round(el, 2) for el in current_state.robot_configuration])
                # print([round(el, 2) for el in reference_trajectory_points[0].robot_configuration])

                try:
                    ocp.solve(x0, x_init, u_init)
                    q_2_traj.append([state[2] for state in ocp.ocp_results.states])
                    # exit(5654)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.plot(
                        [state[2 + 7] for state in ocp.ocp_results.states],
                        label="ocp state vel",
                    )
                    plt.plot(x_init[:, 2 + 7], label="xinit vel")
                    plt.plot(
                        [
                            point.robot_velocity[2]
                            for point in reference_trajectory_points
                        ],
                        label="xtraj vel",
                    )
                    plt.legend()
                    # plt.show()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    q_2_ref = [
                        weighted_trajectory_data[i + j].point.robot_configuration[2]
                        for j in range(HORIZON_SIZE)
                    ]
                    plt.plot(
                        q_2_ref,
                        label="Reference Trajectory",
                        linestyle="--",
                        color="black",
                    )
                    plt.plot(
                        q_2_traj[-1],
                        label="Croco Trajectory",
                        linestyle="-",
                        color="blue",
                    )
                    plt.plot(
                        np.asarray(x_init)[:, 2],
                        label="xinit",
                        linestyle="-",
                        color="red",
                    )
                    # plt.plot(np.asarray(u_init)[:, 2], label="uinit", linestyle="-")
                    plt.legend()
                    plt.show()
                    exit(1)
                except Exception as e:
                    print(
                        f"Failed to solve OCP for effort_weight = {effort_weight}: {e}"
                    )

                # a = pin.aba(
                #     model=rmodel,
                #     data=rdata,
                #     q=current_state.robot_configuration,
                #     v=current_state.robot_velocity,
                #     tau=ocp.ocp_results.feed_forward_terms[0])

                # # # The acceleration is `a`, which contains both joint accelerations
                # # # We can use these accelerations to update the state
                # current_state.robot_configuration += current_state.robot_velocity * DT  # Position update (using Euler integration)
                # current_state.robot_velocity += a * DT  # Velocity update (using Euler integration)

            results[effort_weight] = np.array(q_2_traj)
        q_2_ref = [
            weighted_trajectory_data[i].point.robot_configuration[2]
            for i in range(len(weighted_trajectory_data))
        ]
        for effort_weight, data in results.items():
            alpha = np.linspace(
                1, 0, data.shape[1]
            )  # Linear fade from 1 to 0 (100 rows)

            # Set `up the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # breakpoint()
            # We are going to plot the data with alpha transparency fading over the rows
            # for i in [0]:
            # for i in range(data.shape[1]):
            # ax.scatter(range(i, data.shape[0] + i), data[:, i], color=(0, 0, 1, alpha[i]))  # Blue with decreasing alpha
            ax.scatter(
                range(HORIZON_SIZE), data[0], color=(0, 0, 1, 1.0)
            )  # Blue with decreasing alpha
            print(q_2_ref[0])
            plt.plot(
                range(len(weighted_trajectory_data)),
                q_2_ref,
                label="Reference Trajectory",
                linestyle="--",
                color="black",
            )

            # Customize the plot
            ax.set_title(f"Effort Weight = {effort_weight}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Joint Position (q[2])")

            # Show the plot
            plt.show()
            # breakpoint()

        # # Plot results for the current configuration weight
        # plt.figure(figsize=(10, 6))
        # for effort_weight, q_2_traj in results.items():
        #     plt.plot(q_2_traj, label=f"Effort Weight = {effort_weight}")

        # # Plot reference trajectory

        # plt.plot(q_2_ref, label="Reference Trajectory", linestyle="--", color="black")

        # plt.xlabel("Time Step")
        # plt.ylabel("Joint Position (q[2])")
        # plt.title(f"Joint 2 Trajectory Comparison (Configuration Weight = {configuration_weight})")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
