## Building the docs

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build HTML:

```bash
make html
# Output is in _build/html
```

## Hosting

### Push to ReadTheDocs

By connecting this repository; RTD will run `sphinx-build` automatically.
This repository now packages the Sphinx sources under the `docs/` directory.

To build the docs locally you have several options.

### Using Nix (recommended for reproducible builds):

```bash
# enter a reproducible shell with Sphinx and extensions available
nix build .#packages.x86_64-linux.agimus-controller-doc
```

Then using your favourite web-browser you can visualize the doc:
```bash
firefox result/share/doc/agimus-controller-doc/html
```

### Using Poetry (optional):

```bash
poetry install
poetry run agimus-docs-build
# or use the Makefile inside docs/: cd docs && make html
```

If you don't use Poetry or Nix, install the requirements in `docs/requirements.txt` and run `sphinx-build` manually.
