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
          (_final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (_python-final: python-prev: {
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
            agimus-controller-doc = pkgs.callPackage ./agimus_controller_doc/default.nix { };
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
