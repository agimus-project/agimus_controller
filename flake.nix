{
  description = "Whole Body Model Predictive Control in the AGIMUS architecture";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    systems.follows = "gepetto/systems";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = import inputs.systems;
        imports = [
          inputs.gepetto.flakeModule
          {
            flakoboros.rosOverrideAttrs = {
              agimus-controller = _: _: {
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./agimus_controller
                  ];
                };
              };
              agimus-controller-ros = _: _: {
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
      }
    );
}
