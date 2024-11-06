# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import torch  # noqa: F401

import torchhull

project = torchhull.__package__
version = torchhull.__version__
release = torchhull.__version__
copyright = torchhull.__copyright__  # noqa: A001
author = torchhull.__author__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_defaultargs", # Disabled as it cannot find torchhull's JIT-compiled extension
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

# autodoc_member_order = "bysource"
# autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "special-members": "__init__",
    "undoc-members": True,
}

docstring_default_arg_substitution = "*Default*: "
autodoc_preserve_defaults = True

always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
# intersphinx_disabled_reftypes = ["*"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3


templates_path = ["_templates"]
# exclude_patterns = []

add_module_names = False
coverage_show_missing_items = True

# typehints_defaults = "comma"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# html_css_files = []

html_copy_source = False
