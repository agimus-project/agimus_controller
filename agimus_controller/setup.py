from pathlib import Path
from setuptools import find_packages, setup
from typing import List


PACKAGE_NAME = "agimus_controller"
REQUIRES_PYTHON = ">=3.10.0"


def get_files(dir: Path, pattern: str) -> List[str]:
    return [x.as_posix() for x in (dir).glob(pattern) if x.is_file()]


setup(
    name=PACKAGE_NAME,
    version="0.0.0",
    packages=find_packages(exclude=["tests"]),
    package_dir={f"./{PACKAGE_NAME}": PACKAGE_NAME},
    package_data={"agimus_controller": ["ocp/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        (
            "share/" + PACKAGE_NAME,
            ["package.xml"],
        ),
    ],
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        "crocoddyl",
        "mim_solvers",
        "numpy",
        "pin",
    ],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="Implements whole body MPC in python using the Croccodyl framework.",
    license="BSD-2-Clause",
)
