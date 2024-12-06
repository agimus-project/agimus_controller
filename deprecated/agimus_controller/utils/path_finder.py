import rospkg
from pathlib import Path
import yaml


def get_package_path(package_name) -> Path:
    rospack = rospkg.RosPack()
    return Path(rospack.get_path(package_name))


def get_mpc_params_dict(task_name):
    agimus_demos_description_dir = get_package_path("agimus_demos_description")
    mpc_params_file_path = agimus_demos_description_dir / task_name / "mpc_params.yaml"
    with open(str(mpc_params_file_path), "r") as file:
        return yaml.safe_load(file)
