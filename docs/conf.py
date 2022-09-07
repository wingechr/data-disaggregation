project = "data_disaggregation"
version = "0.6.0"

release = version
html_search_language = "en-US"
html_show_copyright = False
todo_include_todos = False
add_module_names = False
show_authors = True
html_show_sourcelink = False
html_show_sphinx = False
html_theme_options = {}
html_theme = "sphinx_rtd_theme"
master_doc = "index"
source_encoding = "utf-8"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
pygments_style = "sphinx"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["main.css"]
html_js_files = ["main.js"]
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
