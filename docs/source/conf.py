"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
import sys

project = "torchgfn"
copyright = "2022-2025, Joseph Viviano, Sanghyeok Choi, Omar Younis, Victor Schmidt, & Salem Lahlou"
author = "Joseph Viviano, Sanghyeok Choi, Omar Younis, Victor Schmidt, & Salem Lahlou"

sys.path.insert(0, os.path.abspath("../.."))
print("sys.path=", sys.path)


def preprocess_markdown(app, docname, source):
    """Fix paths for different source contexts"""
    if source and len(source) > 0:
        content = source[0]

        # Handle README.md which is at repo root but processed by Sphinx
        if docname == "README":
            # Convert docs/source/path/file.md to path/file.html
            content = re.sub(r"\]\(docs/source/([^)]+)\.md\)", r"](\1.html)", content)

            # Handle any other docs/source/ references
            content = re.sub(r"\]\(docs/source/([^)]+)\)", r"](\1)", content)

            # Handle .github/ paths if you have any
            content = re.sub(r"\]\(\.github/([^)]+)\.md\)", r"](\1.html)", content)

        # Handle files that are already in docs/source/ directory
        else:
            # Just convert .md to .html for relative paths
            content = re.sub(r"\]\(([^)]+)\.md\)", r"](\\1.html)", content)

        source[0] = content


def setup(app):
    app.connect("source-read", preprocess_markdown)


root_doc = "index"
extensions = [
    "myst_parser",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.napoleon",
]
myst_enable_extensions = [
    "colon_fence",
]
source_suffix = {
    ".rst": None,
    ".md": None,
}

autoapi_type = "python"
autoapi_dirs = ["../../src/gfn", "../../tutorials"]
autoapi_member_order = "alphabetical"
autoapi_ignore = ["*tutorials/examples/test_*.py"]

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

html_static_path = ["_static"]
html_theme = "insegel"
html_logo = "logo.png"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 1,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
}
