{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        rosOverrideAttrs = {
          agimus-controller = {
            src = lib.fileset.toSource {
              root = ./.;
              fileset = ./agimus_controller;
            };
          };
          agimus-controller-ros = {
            src = lib.fileset.toSource {
              root = ./.;
              fileset = ./agimus_controller_ros;
            };
          };
        };
      }
    );
}
