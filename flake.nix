{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs = {
    # develop because mim-solvers is not yet available in master
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/develop";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";

    agimus-msgs = {
      url = "github:agimus-project/agimus_msgs/humble-devel";
      inputs.nix-ros-overlay.follows = "nix-ros-overlay";
    };
    colmpc = {
      url = "github:agimus-project/colmpc";
      inputs.nixpkgs.follows = "nix-ros-overlay/nixpkgs";
    };
    linear-feedback-controller-msgs = {
      url = "github:loco-3d/linear-feedback-controller-msgs/humble-devel";
      inputs.nix-ros-overlay.follows = "nix-ros-overlay";
    };
    franka-description = {
      url = "github:agimus-project/franka_description/humble-devel";
      inputs.nix-ros-overlay.follows = "nix-ros-overlay";
    };

    ## Patches for nixpkgs
    # init HPP v6.0.0
    # also: hpp-fcl v2.4.5 -> coal v3.0.0
    patch-hpp = {
      url = "https://github.com/nim65s/nixpkgs/pull/2.patch";
      flake = false;
    };
  };

  outputs =
    {
      agimus-msgs,
      colmpc,
      linear-feedback-controller-msgs,
      franka-description,
      nix-ros-overlay,
      nixpkgs,
      patch-hpp,
      self,
      ...
    }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import ./patched-nixpkgs.nix {
          inherit nixpkgs system;
          overlays = [ nix-ros-overlay.overlays.default ];
          patches = [ patch-hpp ];
        };
      in
      {
        packages = {
          default = self.packages.${system}.agimus-controller-ros;
          agimus-controller = pkgs.python3Packages.callPackage ./agimus_controller/default.nix {
            inherit (colmpc.packages.${system}) colmpc;
            inherit (franka-description.packages.${system}) franka-description;
          };
          agimus-controller-examples =
            pkgs.python3Packages.callPackage ./agimus_controller_examples/default.nix
              {
                inherit (self.packages.${system}) agimus-controller;
                inherit (franka-description.packages.${system}) franka-description;
              };
          agimus-controller-ros = pkgs.python3Packages.callPackage ./agimus_controller_ros/default.nix {
            inherit (self.packages.${system}) agimus-controller;
            inherit (agimus-msgs.packages.${system}) agimus-msgs;
            inherit (franka-description.packages.${system}) franka-description;
            inherit (linear-feedback-controller-msgs.packages.${system}) linear-feedback-controller-msgs;
          };
        };
      }
    );
}
