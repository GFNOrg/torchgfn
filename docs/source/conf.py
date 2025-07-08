# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys


def preprocess_markdown(app, docname, source):
    """Debug and fix markdown paths"""
    if source and len(source) > 0:
        content = source[0]
        original_content = content

        # Log what we're processing (for debugging)
        print(f"Processing docname: {docname}")

        # Handle root-level files
        if docname in ['README', 'CONTRIBUTING'] or '/' not in docname:
            # Fix docs/source/ paths from root files
            content = re.sub(r'\]\(docs/source/([^)#]+)\.md\)', r'](\1.html)', content)
            content = re.sub(r'\]\(docs/source/([^)#]+)\)', r'](\1)', content)
        else:
            # Fix paths from subdirectory files
            content = re.sub(r'\]\(docs/source/([^)#]+)\.md\)', r'](\1.html)', content)
            content = re.sub(r'\]\(\.\./\.\./([^)#]+)\.md\)', r'](\1.html)', content)

        # Clean up any remaining problematic fragments
        content = re.sub(r'#docs/source/', '#', content)
        content = re.sub(r'#$', '', content)  # Remove trailing #

        if content != original_content:
            print(f"Modified content for {docname}")

        source[0] = content


def setup(app):
    app.connect('source-read', preprocess_markdown)


project = "torchgfn"
copyright = "2022-2025, Joseph Viviano, Sanghyeok Choi, Omar Younis, Victor Schmidt, & Salem Lahlou"
author = "Joseph Viviano, Sanghyeok Choi, Omar Younis, Victor Schmidt, & Salem Lahlou"

sys.path.insert(0, os.path.abspath("../.."))
print("sys.path=", sys.path)

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
myst_enable_extensions = [
    "colon_fence",
]

source_suffix = {
    '.rst': None,
    '.md': None,
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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_theme = "alabaster"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
}
