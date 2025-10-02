# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "drrc"
author = "Luk Fleddermann, Gerrit Wellecke"
copyright = f"{date.today().year}, {author}"

sys.path.append("../../Src/drrc")
sys.path.append(".")
import drrc

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


master_doc = "index"

add_module_names = False

modindex_common_prefix = [f"{project}."]
toc_object_entries = False
show_authors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}

html_static_path = ["_static"]


# -- Extension Configuration
autoclass_content = "both"  # include __init__ docstring in class description

# this is in the documentation but isn't used by apidoc ... so probably useless
autodoc_default_options = {
    "undoc-members": False,
}
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
todo_include_todos = True

# Configuration for intersphinx
intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Automate autodoc --------------------------------------------------------
# this generates the full tree of documentation files
from run_autodoc import main

os.environ["SPHINX_APIDOC_OPTIONS"] = "members,no-undoc-members,show-inheritance"

main()
