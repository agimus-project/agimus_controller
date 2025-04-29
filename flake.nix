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
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
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
          _module.args.pkgs = import patchedNixpkgs {
            inherit system;
            overlays = [
              inputs.nix-ros-overlay.overlays.default
              inputs.gepetto.overlays.default
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
                  in
                  prev.rosPackages
                  // {
                    humble = prev.rosPackages.humble.overrideScope (
                      _humble-final: humble-prev: {
                        agimus-controller-ros = humble-prev.agimus-controller-ros.overrideAttrs { inherit src; };
                      }
                    );
                    jazzy = prev.rosPackages.jazzy.overrideScope (
                      _jazzy-final: jazzy-prev: {
                        agimus-controller-ros = jazzy-prev.agimus-controller-ros.overrideAttrs { inherit src; };
                      }
                    );
                  };
              })
            ];
          };
          checks = lib.mapAttrs' (n: lib.nameValuePair "package-${n}") self'.packages;
          packages = {
            default = self'.packages.agimus-controller;
            agimus-controller = pkgs.python3Packages.agimus-controller;
            agimus-controller-examples = pkgs.python3Packages.agimus-controller-examples;
            ros-humble-agimus-controller-examples = pkgs.rosPackages.humble.agimus-controller-ros;
            ros-jazzy-agimus-controller-examples = pkgs.rosPackages.jazzy.agimus-controller-ros;
          };
          treefmt.programs = {
            deadnix.enable = true;
            nixfmt.enable = true;
          };
        };
    };
}
