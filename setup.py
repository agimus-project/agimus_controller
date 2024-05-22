from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['agimus_controller', 'agimus_controller_ros'],
    # scripts=['scripts/myscript'],
    package_dir={
        'agimus_controller_ros':'agimus_controller_ros',
        'agimus_controller':'agimus_controller',
    }
)

setup(**d)
