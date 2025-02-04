from setuptools import find_packages, setup

PACKAGE_NAME = "agimus_controller"
REQUIRES_PYTHON = ">=3.10.0"

setup(
    name=PACKAGE_NAME,
    version="0.0.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        "crocoddyl",
        "mim_solvers",
        "numpy==1.21.5",
        "pinocchio",
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="Implements whole body MPC in python using the Croccodyl framework.",
    license="BSD-2",
    tests_require=["pytest"],
)
