{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    nixpkgs.follows = "gepetto/nixpkgs";
    nix-ros-overlay.follows = "gepetto/nix-ros-overlay";
    treefmt-nix.follows = "gepetto/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      imports = [ inputs.treefmt-nix.flakeModule ];
      perSystem =
        {
          lib,
          pkgs,
          system,
          self',
          ...
        }:
        {
          # Drop this once crocoddyl >= 3.0.1 reaches nix-ros-overlay
          _module.args.pkgs =
            let
              pkgsForPatching = inputs.nixpkgs.legacyPackages.x86_64-linux;
              patchedNixpkgs = (
                pkgsForPatching.applyPatches {
                  inherit (inputs.gepetto) patches;
                  name = "patched nixpkgs";
                  src = inputs.nixpkgs;
                }
              );
            in
            import patchedNixpkgs {
              inherit system;
              overlays = [
                inputs.nix-ros-overlay.overlays.default
                inputs.gepetto.overlays.default
              ];
            };
          checks = lib.mapAttrs' (n: lib.nameValuePair "package-${n}") self'.packages;
          packages =
            let
              src = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions [
                  ./agimus_controller/agimus_controller
                  ./agimus_controller/resource
                  ./agimus_controller/tests
                  ./agimus_controller/package.xml
                  ./agimus_controller/setup.py
                ];
              };
              # src-examples = lib.fileset.toSource {
              #   root = ./.;
              #   fileset = lib.fileset.unions [
              #     ./agimus_controller_examples/agimus_controller_examples
              #     ./agimus_controller_examples/scripts
              #     ./agimus_controller_examples/setup.py
              #   ];
              # };
              src-ros = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions [
                  ./agimus_controller_ros/agimus_controller_ros
                  ./agimus_controller_ros/resource
                  ./agimus_controller_ros/test
                  ./agimus_controller_ros/package.xml
                  ./agimus_controller_ros/setup.cfg
                  ./agimus_controller_ros/setup.py
                ];
              };
            in
            {
              default = self'.packages.agimus-controller;
              agimus-controller = pkgs.python3Packages.agimus-controller.overrideAttrs (oldAttrs: {
                inherit src;
                # Add pytest and any other test dependencies
                nativeBuildInputs = oldAttrs.nativeBuildInputs or [] ++ [
                  pkgs.python3Packages.pytestCheckHook
                ];
                nativeCheckInputs = oldAttrs.nativeCheckInputs or [] ++ [
                  pkgs.python3Packages.pytest
                ];
                # Explicitly define the checkPhase to run pytest
                checkPhase = ''
                  pytest ${src}/agimus_controller/tests
                '';
              });
              # agimus-controller-examples = pkgs.python3Packages.agimus-controller-examples.overrideAttrs {
              #   inherit src-examples;
              # };
              humble-agimus-controller-ros = pkgs.rosPackages.humble.agimus-controller-ros.overrideAttrs (oldAttrs: {
                inherit src-ros;
                  # Add pytest and any other test dependencies
                nativeBuildInputs = oldAttrs.nativeBuildInputs or [] ++ [
                  pkgs.python3Packages.pytestCheckHook
                ];
                nativeCheckInputs = oldAttrs.nativeCheckInputs or [] ++ [
                  pkgs.python3Packages.pytest
                ];
                # Explicitly define the checkPhase to run pytest
                checkPhase = ''
                  pytest ${src}/agimus_controller/tests
                '';
              });
              jazzy-agimus-controller-ros = pkgs.rosPackages.jazzy.agimus-controller-ros.overrideAttrs (oldAttrs: {
                inherit src-ros;
                  # Add pytest and any other test dependencies
                nativeBuildInputs = oldAttrs.nativeBuildInputs or [] ++ [
                  pkgs.python3Packages.pytestCheckHook
                ];
                nativeCheckInputs = oldAttrs.nativeCheckInputs or [] ++ [
                  pkgs.python3Packages.pytest
                ];
                # Explicitly define the checkPhase to run pytest
                checkPhase = ''
                  pytest ${src}/agimus_controller/tests
                '';
              });
            };
          treefmt.programs = {
            deadnix.enable = true;
            nixfmt.enable = true;
          };
        };
    };
}
