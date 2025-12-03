{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    nixpkgs.follows = "gepetto/nixpkgs";
    nix-ros-overlay.follows = "gepetto/nix-ros-overlay";
    systems.follows = "gepetto/systems";
    treefmt-nix.follows = "gepetto/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;
      imports = [ inputs.gepetto.flakeModule ];
      gepetto-pkgs.overlays =
        let
          inherit (inputs.nixpkgs) lib;
        in
        [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                py-force-feedback-mpc = final.python3Packages.buildPythonPackage({
                  pname = "force-feedback-mpc";
                  version = "unstable-2025-12-03";

                  pyproject = false;
                  # None compiling version with newer version of crocoddyl.
                  # TODO package upstream to github:gepetto/nix and fix code.
                  src = final.fetchFromGitHub {
                    owner = "machines-in-motion";
                    repo = "force_feedback_mpc";
                    rev = "04bd43213bc47facd0b752f987fbbdfd3aa5a165";
                    hash = "sha256-mmTtS3a8CqQg17XjZjWd4gaBVuvyt4NSHa7S9VvZKdc=";
                  };

                  nativeBuildInputs = [
                    final.cmake
                    final.pkg-config
                  ];

                  propagatedBuildInputs = [
                    final.eigen
                    final.jrl-cmakemodules
                    final.llvmPackages.openmp
                    python-final.boost
                    python-final.eigenpy
                    python-final.example-robot-data
                    python-final.pinocchio
                    python-final.crocoddyl
                    python-final.pythonImportsCheckHook
                  ];

                  cmakeFlags = [
                    (lib.cmakeBool "BUILD_PYTHON_INTERFACE" true)
                  ];

                  doCheck = true;
                  pythonImportsCheck = [ "force_feedback_mpc" ];

                  meta = {
                    description = "Optimal control tools to achieve force feedback control in MPC ";
                    homepage = "https://github.com/machines-in-motion/force_feedback_mpc";
                    license = lib.licenses.bsd2;
                    maintainers = with lib.maintainers; [ nim65s ];
                    platforms = lib.platforms.unix;
                  };
                  patchPhase = ''
                    # Update cmake_minimum_required version
                    sed -i 's/cmake_minimum_required(VERSION 3\.10)/cmake_minimum_required(VERSION 3.22)/' CMakeLists.txt

                    # Replace the submodule block
                    sed -i '/# Check if the submodule cmake have been initialized/{
                      N;N;N;N;N;N;N;N;N;N;N;N
                      c\
                  find_package(jrl-cmakemodules REQUIRED CONFIG)\
                  get_property(\
                        JRL_CMAKE_MODULES\
                        TARGET jrl-cmakemodules::jrl-cmakemodules\
                        PROPERTY INTERFACE_INCLUDE_DIRECTORIES\
                      )\
                  message(STATUS "JRL cmakemodules found on system at \$\{JRL_CMAKE_MODULES\}")
                    }' CMakeLists.txt

                    echo "==== CMakeLists.txt after patch ===="
                    cat CMakeLists.txt
                  '';
                });
                agimus-controller = python-prev.agimus-controller.overrideAttrs {
                  src = lib.fileset.toSource {
                    root = ./.;
                    fileset = lib.fileset.unions [
                      ./agimus_controller
                    ];
                  };
                  # Remove when the CI passes. And update https://github.com/Gepetto/nix
                  nativeCheckInputs = [
                    prev.python3Packages.pytest
                    prev.python3Packages.pytestCheckHook
                  ];
                  propagatedBuildInputs = python-prev.agimus-controller.propagatedBuildInputs ++ [
                    python-final.py-force-feedback-mpc
                  ];
                  doCheck = true;
                };
                agimus-controller-examples = python-prev.agimus-controller-examples.overrideAttrs {
                  src = lib.fileset.toSource {
                    root = ./.;
                    fileset = lib.fileset.unions [
                      ./agimus_controller_examples
                    ];
                  };
                };
              })
            ];
            rosPackages =
              let
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./agimus_controller_ros
                  ];
                };
                nativeCheckInputs = [
                  prev.python3Packages.pytest
                  prev.python3Packages.pytestCheckHook
                ];
                doCheck = true;
              in
              prev.rosPackages
              // {
                humble = prev.rosPackages.humble.overrideScope (
                  _humble-final: humble-prev: {
                    agimus-controller-ros = humble-prev.agimus-controller-ros.overrideAttrs {
                      inherit
                        src
                        doCheck
                        nativeCheckInputs
                        ;
                    };
                  }
                );
                jazzy = prev.rosPackages.jazzy.overrideScope (
                  _jazzy-final: jazzy-prev: {
                    agimus-controller-ros = jazzy-prev.agimus-controller-ros.overrideAttrs {
                      inherit
                        src
                        doCheck
                        nativeCheckInputs
                        ;
                    };
                  }
                );
              };
          })
        ];
      perSystem =
        {
          lib,
          pkgs,
          self',
          ...
        }:
        {
          devShells = {
            default = self'.devShells.env;
            env = pkgs.mkShell {
              name = "env";
              packages = [
                (pkgs.python3.withPackages (p: [
                  p.agimus-controller
                  p.agimus-controller-examples
                  p.pytest
                ]))
              ];
            };
          }
          # 2025/11/07 Remove this env from MacOs as franka_description is
          # written as broken
          // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            ros-env = pkgs.mkShell {
              name = "ros-env";
              packages = [
                self'.packages.ros-env
              ];
            };
          };
          packages = {
            default = self'.packages.agimus-controller;
            agimus-controller = pkgs.python3Packages.agimus-controller;
            agimus-controller-examples = pkgs.python3Packages.agimus-controller-examples;
            py-force-feedback-mpc = pkgs.python3Packages.py-force-feedback-mpc;
          }
          // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            ros-humble-agimus-controller-ros = pkgs.rosPackages.humble.agimus-controller-ros;
            ros-jazzy-agimus-controller-ros = pkgs.rosPackages.jazzy.agimus-controller-ros;
          }
          // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            ros-env =
              with pkgs.rosPackages.jazzy;
              buildEnv {
                name = "ros-env";
                paths = [
                  pkgs.python3Packages.meshcat
                  pkgs.python3Packages.coal
                  pkgs.python3Packages.pinocchio
                  pkgs.python3Packages.gepetto-gui
                  pkgs.python3Packages.agimus-controller
                  pkgs.rosPackages.humble.franka-description
                  pkgs.rosPackages.humble.xacro
                  pkgs.rosPackages.humble.ament-index-python
                  pkgs.rosPackages.humble.realsense2-description
                ];
              };
          };
        };
    };
}
