from os import environ
from pathlib import Path
import sys

import numpy as np
import yaml
from pinocchio.visualize import MeshcatVisualizer
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels

import xacro
from ament_index_python.packages import get_package_share_directory

FRANKA_DESCRIPTION_PATH = Path(get_package_share_directory("franka_description"))
print(environ["AMENT_PREFIX_PATH"])
print(FRANKA_DESCRIPTION_PATH)
environ["AMENT_PREFIX_PATH"] += ":" + str(FRANKA_DESCRIPTION_PATH)
print(environ["AMENT_PREFIX_PATH"])

# Load the example robot model using example robot data to get the URDF path.
srdf_path = FRANKA_DESCRIPTION_PATH / "robots" / "fer" / "fer.srdf"
with open(srdf_path, "r") as srdf_file:
    srdf_xml = srdf_file.read()
robot_xacro_path = str(FRANKA_DESCRIPTION_PATH / "robots" / "fer" / "fer.urdf.xacro")
env_xacro_path = Path(__file__).parent / "resources" / "environment.xacro"
params_path = str(Path(__file__).parent / "resources" / "agimus_controller_params.yaml")
with open(params_path, "r") as file:
    mpc_params = yaml.safe_load(file)["agimus_controller_node"]["ros__parameters"]
robot_urdf_xml = xacro.process_file(
    robot_xacro_path,
    mappings={"with_sc": "true"},
).toxml()
env_urdf_xml = xacro.process_file(env_xacro_path).toxml()

urdf_path = robot_urdf_xml
env_urdf = env_urdf_xml
# env_urdf = None
srdf_path = srdf_path
urdf_meshes_dir = FRANKA_DESCRIPTION_PATH
free_flyer = False
moving_joint_names = [
    "fer_joint1",
    "fer_joint2",
    "fer_joint3",
    "fer_joint4",
    "fer_joint5",
    "fer_joint6",
    "fer_joint7",
    "fer_finger_joint1",
    "fer_finger_joint2",
]
reduced_nq = len(moving_joint_names)
params = RobotModelParameters(
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

robot_models = RobotModels(params)


# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in
# a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz = MeshcatVisualizer(
        robot_models.robot_model,
        robot_models.collision_model,
        robot_models.visual_model,
    )
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. "
        "It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()
viz.displayVisuals(True)
viz.displayCollisions(True)
viz.displayFrames(False)

# Display a robot configuration.
q0 = np.array(
    [
        -0.3619834760502907,
        -1.3575006398318104,
        0.969610481368033,
        -2.6028532848927295,
        0.2040785081450368,
        1.9436352693107668,
        0.6423896937386857,
        0.02,
        0.02,
    ]
)
viz.display(q0)
