# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys, os
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'NC-OPT'
copyright = '2020, Weiwei Kong'
author = 'Weiwei Kong'

# The full version, including alpha/beta/rc tags
release = '0.1a1'
# The short version
version = '0.1a1'


# -- General configuration ---------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.viewcode', 
	'sphinxcontrib.matlab', 
	'sphinx.ext.autodoc', 
	'sphinx.ext.napoleon',
	'sphinx_rtd_theme', 
	'sphinx.ext.mathjax', 
	'sphinx_math_dollar'
	]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
matlab_src_dir = os.path.abspath('..')
primary_domain = 'mat'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The master doc.
master_doc = 'index'

# Autodoc options.
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_title = "NC-OPT User Guide"
html_short_title = "User Guide"
html_logo = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# --- LaTeX options ----------------------------------------------------------

