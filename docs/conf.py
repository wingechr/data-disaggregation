# coding=utf-8
import os
import sys

# import package (from setup) to get infos
# add root dir to python path (for tools lke nbsphinx)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
os.environ["PYTHONPATH"] = BASE_DIR
sys.path.insert(0, BASE_DIR)  # needed for import setup

import data_disaggregation as pkg  # noqa: E402

project = pkg.__title__
description = pkg.__doc__
version = pkg.__version__
author = pkg.__author__
email = pkg.__email__
copyright = pkg.__copyright__

release = version
html_search_language = "en"
html_show_copyright = False
todo_include_todos = False
add_module_names = False
show_authors = True
html_show_sourcelink = False
html_show_sphinx = True
docs_path = "."
html_theme_options = {}
html_theme = "sphinx_rtd_theme"
master_doc = "index"
source_encoding = "utf-8"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

pygments_style = "sphinx"
html_logo = os.path.join(docs_path, "_static/logo.svg")
html_favicon = os.path.join(docs_path, "_static/favicon.ico")
templates_path = [os.path.join(docs_path, "_templates")]
html_static_path = [os.path.join(docs_path, "_static")]
exclude_dirs = []  # do not include in autodoc
nitpicky = False
html_use_index = True
add_function_parentheses = True


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "myst_parser",  # markdown
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Mathjax settings
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js"
mathjax_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/CommonHTML"],
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    },
    "HTML-CSS": {"availableFonts": ["TeX"]},
    "menuSettings": {"zoom": "Double-Click", "mpContext": True, "mpMouse": True},
    "config": [],
    "showProcessingMessages": False,
    "messageStyle": "none",
    "showMathMenu": False,
    "displayAlign": "left",
}

# graphviz
graphviz_output_format = "svg"  # svg | png
