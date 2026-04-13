import os
import sys
from datetime import datetime

# Compute and add repository root to sys.path so autodoc can import
# the `agimus_controller` package regardless of current working dir.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# When building docs in CI or on developer machines the environment may not
# have heavy C++-backed dependencies like `crocoddyl`, `pinocchio` or
# `colmpc`. Mock them so Sphinx can import the Python modules and
# autosummary/autodoc can generate API pages.
# Note: Now installed via nix, so no need to mock them anymore.
# autodoc_mock_imports = [
#     "crocoddyl",
#     "pinocchio",
#     "colmpc",
# ]

project = "agimus_controller"
author = "agimus_controller contributors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# MyST-Parser configuration to enable Sphinx directives in markdown
myst_enable_extensions = [
    "colon_fence",  # Enable ::: for directives
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx: link to Python and NumPy docs
# Disable remote intersphinx inventory fetches in offline/CI builds to avoid
# noisy network warnings. If you need cross-references to Python/NumPy docs
# re-enable the mapping below.
intersphinx_mapping = {}
