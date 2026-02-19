{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs = {
    gazebros2nix.url = "github:gepetto/gazebros2nix";
    flake-parts.follows = "gazebros2nix/flake-parts";
    nixpkgs.follows = "gazebros2nix/nixpkgs";
    nix-ros-overlay.follows = "gazebros2nix/nix-ros-overlay";
    systems.follows = "gazebros2nix/systems";
    treefmt-nix.follows = "gazebros2nix/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = [ "x86_64-linux" ];
        imports = [
          inputs.gazebros2nix.flakeModule
          {
            gazebros2nix.rosPackages = {
              agimus-controller = _final: _ros-final: {
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./agimus_controller
                  ];
                };
              };

              # agimus-controller-examples = _final: _ros-final: {
              #   src = lib.fileset.toSource {
              #     root = ./.;
              #     fileset = lib.fileset.unions [
              #       ./agimus_controller_examples
              #     ];
              #   };
              # };

              agimus-controller-ros = _final: _ros-final: {
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./agimus_controller_ros
                  ];
                };
              };

            };
          }
        ];
        perSystem =
          { self', ... }:
          {
            packages.default = self'.packages.ros-rolling-agimus-controller-ros;
          };
      }
    );
}
