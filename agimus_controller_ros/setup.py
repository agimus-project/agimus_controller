from pathlib import Path
from typing import List

from setuptools import setup

from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = "agimus_controller_ros"
project_source_dir = Path(__file__).parent


for module_name, yaml_file in [
    (
        "agimus_controller_parameters",
        "agimus_controller_ros/agimus_controller_parameters.yaml",
    ),
    (
        "trajectory_weights_parameters",
        "agimus_controller_ros/trajectory_weights_parameters.yaml",
    ),
]:
    generate_parameter_module(module_name, yaml_file)


def get_files(dir: Path, pattern: str) -> List[str]:
    return [x.as_posix() for x in (dir).glob(pattern) if x.is_file()]


setup(
    name=package_name,
    version="0.0.0",
    packages=["agimus_controller_ros"],
    package_dir={},
    install_requires=["setuptools"],
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            f"share/{package_name}/launch",
            get_files(project_source_dir / "launch", "*.launch.py"),
        ),
    ],
    maintainer="Guilhem Saurel",
    maintainer_email="gsaurel@laas.fr",
    description="ROS2 agimus_controller package",
    license="BSD",
    entry_points={
        "console_scripts": [
            "simple_trajectory_publisher = agimus_controller_ros.simple_trajectory_publisher:main",
            "trajectory_publisher_with_visual_servoing = agimus_controller_ros.trajectory_publisher_with_visual_servoing:main",
            "agimus_controller_node = agimus_controller_ros.agimus_controller:main",
            "mpc_debugger_node = agimus_controller_ros.mpc_debugger_node:main",
        ],
    },
    tests_require=["pytest"],
)
