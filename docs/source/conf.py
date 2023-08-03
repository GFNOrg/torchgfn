# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = "torchgfn"
copyright = "2022-2023, Salem Lahlou, Joseph Viviano, & Victor Schmidt"
author = "Salem Lahlou, Joseph Viviano, & Victor Schmidt"


sys.path.insert(0, os.path.abspath("../.."))
print(sys.path)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
root_doc = "index"
extensions = [
    "myst_parser",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

autoapi_type = "python"
autoapi_dirs = ["../../src/gfn", "../../tutorials"]
autoapi_member_order = "alphabetical"

autodoc_typehints = "description"
mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
mathjax3_config = {
    "tex": {
        "inlineMath": [
            ["$", "$"],
            ["\\(", "\\)"],
        ],
        "processEscapes": True,
    },
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_theme = "renku"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
