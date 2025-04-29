{
  agimus-controller,
  buildPythonPackage,
  franka-description,
  hpp-corbaserver,
  hpp-gepetto-viewer,
  hpp-manipulation-corba,
  lib,
  meshcat,
  matplotlib,
  numpy,
  pinocchio,
  #pytestCheckHook, # Uncomment to add tests.
  setuptools,
}:
buildPythonPackage {
  pname = "agimus-controller-examples";
  version = "0-unstable-2025-01-15";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./agimus_controller_examples
      ./setup.py
      # ./tests # Uncomment to add tests.
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    agimus-controller
    franka-description
    hpp-corbaserver
    hpp-manipulation-corba
    hpp-gepetto-viewer
    meshcat
    matplotlib
    numpy
    pinocchio
  ];

  #nativeCheckInputs = [ pytestCheckHook ]; # Uncomment to add tests.
  doCheck = true;
  pythonImportsCheck = [ "agimus_controller_examples" ];
  dontUseCmakeConfigure = true; # Something is propagating cmake…
  dontWrapQtApps = true;


  meta = {
    description = "agimus_controller_examples is a sandbox for the agimus_controller package.";
    homepage = "https://github.com/agimus-project/agimus_controller";
    license = lib.licenses.bsd3;
    maintainers = [ lib.maintainers.nim65s ];
    platforms = lib.platforms.linux;
  };
}
