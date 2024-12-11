from setuptools import setup

package_name = "agimus_controller_examples"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name]
    + [
        package_name + "." + pkg
        for pkg in ["hpp_panda", "main", "main/panda", "main/ur3", "resources", "utils"]
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="Set of Examples using Agimus_controller package",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
