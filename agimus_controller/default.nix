{
  buildPythonPackage,
  colmpc,
  crocoddyl,
  coal,
  example-robot-data,
  franka-description,
  lib,
  mim-solvers,
  numpy,
  pinocchio,
  pip,
  pytestCheckHook,
  rosPackages,
  rospkg,
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller
      ./package.xml
      ./resource
      ./setup.py
      ./tests
    ];
  };

  build-system = [ setuptools pip ];

  dependencies = [
    colmpc
    crocoddyl
    coal
    example-robot-data
    mim-solvers
    numpy
    pinocchio
    franka-description
    rosPackages.humble.xacro
    rosPackages.humble.ament-index-python
    rospkg
  ];

  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  pythonImportsCheck = [ "agimus_controller" ];
  dontWrapQtApps = true;
  dontUseCmakeConfigure = true; # Something is propagating cmake…

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
