from setuptools import setup

package_name = "agimus_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name]
    + [
        package_name + "." + pkg
        for pkg in ["ocps", "robot_model", "utils", "visualization"]
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description=" The agimus_controller package",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
