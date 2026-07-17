# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Configuration file for the Sphinx documentation builder.
#
# Docs are generated from the Python docstrings and (when the compiled CUDA
# extension is importable) the pybind11 binding docstrings. See the full options
# at https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the in-tree sources importable without installing/building the package,
# so the docs build on machines without a CUDA toolchain (e.g. Read the Docs).
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "gpusim2grid"
copyright = "RTE (https://www.rte-france.com)"
author = "DONNOT Benjamin"

try:
    import gpusim2grid
    release = gpusim2grid.__version__
except Exception:  # pragma: no cover - best effort
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy / Google style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",              # allow Markdown sources (README, DISCLAIMER)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc -----------------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Heavy / platform-specific dependencies that are not needed to read the API.
autodoc_mock_imports = [
    "torch",
    "lightsim2grid",
    "pandapower",
    "numba",
]

# The compiled CUDA extension cannot be built on a CPU-only docs runner. If it
# is importable (a local build on a CUDA machine) its pybind11 docstrings are
# included; otherwise it is mocked so the pure-Python API still builds.
try:
    import gpusim2grid._gpusim2grid  # noqa: F401
except Exception:
    autodoc_mock_imports.append("gpusim2grid._gpusim2grid")

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
