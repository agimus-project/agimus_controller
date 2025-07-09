import os
import yaml

import pickle
from pathlib import Path
from ament_index_python import get_package_share_directory
from agimus_controller_examples.utils.read_from_bag_trajectory import (
    save_rosbag_outputs_to_pickle,
)

from agimus_controller.plots.plots_utils import plot_mpc_data

from agimus_controller_examples.utils.set_models_and_mpc import get_panda_models


with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)

bag_file_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])
picke_file_path = mpc_config["bag_directory"] + "pickle_" + mpc_config["bag_name"]
save_rosbag_outputs_to_pickle(bag_file_path, picke_file_path)
with open(picke_file_path, "rb") as pickle_file:
    mpc_data = pickle.load(pickle_file)

config_folder_path = (
    Path(get_package_share_directory("agimus_demo_03_mpc_dummy_traj")) / "config"
)
env_xacro_path = (
    Path(get_package_share_directory("agimus_demo_05_pick_and_place"))
    / "urdf"
    / "environment.urdf.xacro"
)
robot_models = get_panda_models(config_folder_path, env_xacro_path)
rmodel = robot_models.robot_model
which_plots = [
    "computation_time",
    "collision_distance",
    "iter",
    "visual_servoing",
    "predictions",
]
plot_mpc_data(mpc_data, mpc_config, rmodel, which_plots)
