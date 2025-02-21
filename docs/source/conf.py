# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
print("Sphinx sys.path:", sys.path)
sys.path.insert(0, os.path.abspath('../'))  # Ensure modules are found

# Add the root directory to sys.path (or other directories as needed)
sys.path.insert(0, os.path.abspath('../solvers'))
sys.path.insert(0, os.path.abspath('../network'))

project = 'Neural Operator'
copyright = '2025, Holland'
author = 'Holland'
release = '0.0'

extensions = [
    'sphinx.ext.autodoc',  # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support Google and NumPy-style docstrings
    'sphinx.ext.viewcode'   # Include source code links
]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

autodoc_mock_imports = ["torch"]

