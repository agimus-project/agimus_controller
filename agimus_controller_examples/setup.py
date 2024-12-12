from setuptools import find_packages, setup

package_name = "agimus_controller_examples"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guilhem Saurel",
    maintainer_email="guilhem.saurel@laas.fr",
    description="Example package using visualizers like meshcat and matplotlib to debug the agimus_controller MPCs",
    license="BSD-2",
    tests_require=["pytest"],
)
