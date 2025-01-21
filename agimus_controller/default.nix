{
  buildPythonPackage,
  colmpc,
  crocoddyl,
  coal,
  lib,
  matplotlib,
  meshcat,
  mim-solvers,
  numpy,
  pinocchio,
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
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    colmpc
    crocoddyl
    coal
    matplotlib
    meshcat
    mim-solvers
    numpy
    pinocchio
    rosPackages.humble.xacro
    rospkg
  ];

  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  pythonImportsCheck = [ "agimus_controller" ];

  meta = {
    description = "The agimus_controller package";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
