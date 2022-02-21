# coding=utf-8
import os

project = "Data Disaggregation"
description = "TODO SHORT DESCRIPTION"
version = "0.1.0"
author = "Christian Winger"
email = "c.winger@oeko.de"
copyright = "GPLv3+"
urls = {"code": "https://github.com/wingechr/data-disaggregation"}

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
source_suffix = [".rst", ".md"]


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
    # sphinxcontrib
    "sphinxcontrib.mermaid",
    # TODO: still not working with python 3.10: https://github.com/sphinx-contrib/napoleon/issues/9
    # "sphinxcontrib.napoleon",  # requires sphinxcontrib-napoleon
    "m2r2",  # new md -> rst
    # "sphinx_click",  # requires sphinx-click
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


# fix warnings about docstring referencing builtin types
nitpick_ignore = [
    #    ("py:obj", "int"),
    #    ("py:obj", "str"),
    #    ("py:obj", "bool"),
    #    ("py:obj", "optional"),
]
