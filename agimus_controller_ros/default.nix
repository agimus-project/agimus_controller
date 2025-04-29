{
  agimus-controller,
  agimus-msgs,
  buildPythonPackage,
  franka-description,
  lib,
  linear-feedback-controller-msgs,
  numpy,
  pinocchio,
  pytestCheckHook,
  python,
  rosPackages,
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller-ros";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller_ros
      ./package.xml
      ./resource
      ./setup.py
      ./test
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    agimus-controller
    agimus-msgs
    franka-description
    linear-feedback-controller-msgs
    numpy
    pinocchio
    rosPackages.humble.builtin-interfaces
    rosPackages.humble.generate-parameter-library-py
    rosPackages.humble.geometry-msgs
    rosPackages.humble.rclpy
    rosPackages.humble.std-msgs
  ];

  doCheck = true;
  # Ensure ROS environment variables are set before running pytest
  preCheck = ''
    mkdir -p $TMPDIR/ros_logs
    chmod 777 $TMPDIR/ros_logs
    export ROS_LOG_DIR=$TMPDIR/ros_logs
    export ROS_HOME=$TMPDIR/ros_home
    export PYTHONPATH=$PYTHONPATH:${lib.getLib rosPackages.humble.rclpy}/${python.sitePackages}
    export AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH:${lib.getLib rosPackages.humble.rclpy}
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:${lib.getLib rosPackages.humble.rclpy}
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  '';
  nativeCheckInputs = [
    pytestCheckHook
    rosPackages.humble.rclpy
    rosPackages.humble.ament-copyright
    rosPackages.humble.ament-flake8
    rosPackages.humble.ament-pep257
  ];
  pythonImportsCheck = [ "agimus_controller_ros" ];
  dontUseCmakeConfigure = true; # Something is propagating cmake…
  dontWrapQtApps = true;

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
