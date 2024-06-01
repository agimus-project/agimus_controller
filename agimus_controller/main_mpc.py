import numpy as np

import pybullet
from mim_robots.pybullet.env import BulletEnvWithGround

from ocp import OCPPandaReachingColWithMultipleCol
from wrapper_panda import PandaRobot
from scenes import Scene
from mpc import MPC

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ###
# # # # # # # # # # # # # # # # # # #

# Name of the scene (can be changed by "ball" and "wall")
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

# Parameters of the OCP
max_iter = 4  # Maximum iterations of the solver
max_qp_iters = (
    25  # Maximum iterations for solving each qp solved in one iteration of the solver
)
dt = 2e-2
T = 10
WEIGHT_GRIPPER_POSE = 1e2
WEIGHT_GRIPPER_POSE_TERM = 1e2
WEIGHT_xREG = 1e-2
WEIGHT_xREG_TERM = 1e-2
WEIGHT_uREG = 1e-4
max_qp_iters = 25
callbacks = False
safety_threshhold = 7e-2

# Parameters of the MPC
T_sim = 0.5

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
    T_sim=T_sim,
)
mpc.solve()
mpc.plot_collision_distances()
mpc.plot_mpc_results()
