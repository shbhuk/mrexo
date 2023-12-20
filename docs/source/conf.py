# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../mrexo'))  # Source code dir relative to this file



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MRExo'
copyright = '2023, Shubham Kanodia and Matthias Yang He'
author = 'Shubham Kanodia, Matthias Yang He'
release = 'v1.1.5'



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.mathjax',
   'sphinx.ext.napoleon',
   'sphinx_toolbox.installation',
   'sphinx_toolbox.collapse',
   'sphinx_rtd_theme',
]
autosummary_generate = True
add_module_names = False # whether to prepend module names to functions/objects
autodoc_member_order = 'bysource' # sort docs for members by the order in which they appear in the module; default is 'alphabetical'

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = [
   "numpy",
   "matplotlib",
   "scipy",
   "mpl_toolkits",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = [] # ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = ""
