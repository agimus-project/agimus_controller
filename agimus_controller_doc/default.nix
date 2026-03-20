{
  stdenv,
  python3Packages,
  lib,
  agimus-controller,
  ...
}:

stdenv.mkDerivation rec {
  pname = "agimus-controller-doc";
  version = "0.1.0";

  src = ./.;

  buildInputs = with python3Packages; [
    sphinx
    sphinx-rtd-theme
    myst-parser
    sphinx-autodoc-typehints
    agimus-controller
  ];

  buildPhase = ''
    mkdir -p _build/html
    sphinx-build -b html ${src}/docs _build/html
  '';

  installPhase = ''
    mkdir -p $out/share/doc/${pname}
    cp -r _build/html $out/share/doc/${pname}/html
  '';

  meta = with lib; {
    description = "Sphinx HTML documentation for agimus_controller";
    license = licenses.unfree; # docs only
    platforms = platforms.unix;
  };
}
