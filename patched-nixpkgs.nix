{
  nixpkgs,
  overlays ? [ ],
  patches ? [ ],
  system ? "x86_64-linux",
  ...
}:
let
  # bare nixpkgs to import the `applyPatches` function
  super = import nixpkgs { inherit system; };
in
import (super.applyPatches {
  inherit patches;
  name = "patched nixpkgs";
  src = nixpkgs;
}) { inherit overlays system; }
