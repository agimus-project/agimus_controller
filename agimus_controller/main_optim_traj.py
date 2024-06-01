import time
import numpy as np
import pinocchio as pin

from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
from ocp import OCPPandaReachingColWithMultipleCol
from scenes import Scene

### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.01

# Creating the robot
robot_wrapper = PandaWrapper(auto_col=True, capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

# Creating the scene
scene = Scene("wall")
rmodel1, cmodel1, TARGET1, TARGET2, q0 = scene.create_scene_from_urdf(rmodel, cmodel)
# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET2,
    robot_model=rmodel1,
    robot_collision_model=cmodel1,
    robot_visual_model=vmodel,
)

### INITIAL X0
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM WITHOUT WARM START
problem = OCPPandaReachingColWithMultipleCol(
    rmodel1,
    cmodel1,
    TARGET2,
    T,
    dt,
    x0,
    WEIGHT_GRIPPER_POSE=100,
    WEIGHT_xREG=1e-2,
    WEIGHT_uREG=1e-4,
    SAFETY_THRESHOLD=2.5e-3,
    callbacks=True,
)

ddp = problem()

XS_init = [x0] * (T + 1)
# US_init = [np.zeros(rmodel.nv)] * T
US_init = ddp.problem.quasiStatic(XS_init[:-1])
# Solving the problem
ddp.solve(XS_init, US_init)

print("End of the computation, press enter to display the traj if requested.")
### DISPLAYING THE TRAJ
while True:
    vis.display(q0)
    input()
    for xs in ddp.xs:
        vis.display(np.array(xs[:7].tolist()))
        time.sleep(1e-1)
    input()
    print("replay")
