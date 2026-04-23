from pathlib import Path
from typing import List
import os

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

from generate_parameter_library_py.setup_helper import generate_parameter_module
from generate_parameter_library_py.generate_python_module import (
    run as generate_python_module,
)

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
        # Generate the parameter modules into the package directory inside
        # the build output. Compute the workspace `install` path from
        # `self.build_lib` and pass it to the helper with `merge_install=True`
        # so the generator only writes the merged `install/lib/.../site-packages`
        # layout and not a per-package `install/<pkg>/...` layout.
        target = Path(self.build_lib) / package_name
        build_lib_path = Path(self.build_lib)
        colcon_ws = None
        for p in build_lib_path.parents:
            if p.name == "build":
                colcon_ws = p.parent
                break
        if colcon_ws is None:
            colcon_ws = Path.cwd()
        target.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        try:
            os.chdir(target)
            for module_name, yaml_file in parameter_modules:
                yaml_path = project_source_dir / yaml_file
                # Generate only the build/lib copy directly using the
                # generator entrypoint to avoid the helper also writing
                # into multiple install locations.
                target_file = Path(target) / (module_name + ".py")
                generate_python_module(str(target_file), str(yaml_path), "")
        finally:
            os.chdir(cwd)
        super().run()


class develop(_develop):
    def run(self):
        # In editable/develop (used by symlink-install) generate the
        # parameter modules into the workspace `install` directory so
        # generated modules live in `install/.../site-packages` instead
        # of polluting the source tree. Respect `--merge-install` if
        # present on the command line.
        # Determine the colcon workspace root by locating the ancestor
        # directory that contains a `src` directory.
        colcon_ws = None
        for p in project_source_dir.parents:
            if (p / "src").is_dir():
                colcon_ws = p
                break
        if colcon_ws is None:
            colcon_ws = Path.cwd()
        install_base = str(colcon_ws / "install")
        # Treat `--symlink-install` as a request to place generated
        # modules in the merged install layout too, so generated files
        # live under `install/lib/.../site-packages` rather than in src.
        argv_str = " ".join(os.sys.argv)
        merge_install = ("--merge-install" in argv_str) or (
            "--symlink-install" in argv_str
        )
        for module_name, yaml_file in parameter_modules:
            yaml_path = project_source_dir / yaml_file
            # Use the helper so it writes the install copy in the
            # requested install layout.
            generate_parameter_module(
                module_name,
                str(yaml_path),
                install_base=install_base,
                merge_install=merge_install,
            )
        super().run()


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    package_dir={},
    install_requires=["setuptools", "generate_parameter_library_py"],
    zip_safe=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
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
