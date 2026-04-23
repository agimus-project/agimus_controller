from pathlib import Path
from typing import List
import os

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = "agimus_controller_ros"
project_source_dir = Path(__file__).parent

# List of (module_name, yaml_file) to generate. We will generate these
# into the build output during the `build_py` command so the source tree
# is not polluted with generated files.
parameter_modules = [
    (
        "agimus_controller_parameters",
        "agimus_controller_ros/agimus_controller_parameters.yaml",
    ),
    (
        "trajectory_weights_parameters",
        "agimus_controller_ros/trajectory_weights_parameters.yaml",
    ),
]


def get_files(dir: Path, pattern: str) -> List[str]:
    return [x.as_posix() for x in (dir).glob(pattern) if x.is_file()]


class build_py(_build_py):
    def run(self):
        # ensure target package dir in build output exists
        target = Path(self.build_lib) / "agimus_controller_ros"
        target.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        try:
            # run generator with cwd set to the build package dir so generated
            # modules are written into the build output (and later installed
            # under site-packages), not into the source tree.
            os.chdir(target)
            for module_name, yaml_file in parameter_modules:
                yaml_path = project_source_dir / yaml_file
                generate_parameter_module(module_name, str(yaml_path))
        finally:
            os.chdir(cwd)
        super().run()


class develop(_develop):
    def run(self):
        # In editable/develop (used by symlink-install) generate the
        # parameter modules into the source package directory so the
        # symlinked install will expose them.
        target = project_source_dir / "agimus_controller_ros"
        target.mkdir(parents=True, exist_ok=True)
        for module_name, yaml_file in parameter_modules:
            yaml_path = project_source_dir / yaml_file
            generate_parameter_module(module_name, str(yaml_path))
        super().run()


setup(
    name=package_name,
    version="0.0.0",
    packages=["agimus_controller_ros"],
    package_dir={},
    install_requires=["setuptools", "generate_parameter_library_py"],
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    maintainer="Guilhem Saurel",
    maintainer_email="gsaurel@laas.fr",
    description="ROS2 agimus_controller package",
    license="BSD-2-Clause",
    entry_points={
        "console_scripts": [
            "simple_trajectory_publisher = agimus_controller_ros.simple_trajectory_publisher:main",
            "agimus_controller_node = agimus_controller_ros.agimus_controller:main",
            "mpc_debugger_node = agimus_controller_ros.mpc_debugger_node:main",
            "mpc_plot_node = agimus_controller_ros.mpc_plot_node:main",
        ],
    },
    cmdclass={"build_py": build_py, "develop": develop},
)
