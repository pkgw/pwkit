# -*- coding: utf-8 -*-

import sphinx_rtd_theme

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
]

project = "pwkit"
release = "1.2.2"  # cranko project-version
version = ".".join(release.split(".")[:2])

copyright = "2015-2023, Peter K. G. Williams and collaborators"
author = "Peter K. G. Williams and collaborators"

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = []

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

pygments_style = "sphinx"
todo_include_todos = False


# Intersphinx

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://data.astropy.org/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://docs.scipy.org/doc/numpy/",
        (None, "http://data.astropy.org/intersphinx/numpy.inv"),
    ),
    "astropy": ("http://docs.astropy.org/en/stable/", None),
}


# HTML output settings

html_theme = "sphinx_rtd_theme"
# html_theme_options = {}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]
htmlhelp_basename = "pwkitdoc"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []


# Tomfoolery to fake modules that readthedocs.org doesn't know. We need to do
# a super-duper hack for `pwkit.sherpa` because of how multiple inheritance
# interacts with the Mock object system.

import sys
from mock import Mock as MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        if name in ("ArithmeticModel", "XSAdditiveModel"):
            return dict
        if name == "CompositeModel":
            return Mock
        return Mock()


sys.modules.update(
    (m, Mock())
    for m in [
        "cairo",
        "gi",
        "gi.repository",
        "glib",
        "gtk",
        "sherpa",
        "sherpa.astro",
        "sherpa.astro.ui",
        "sherpa.astro.xspec",
        "sherpa.astro.xspec._xspec",
        "sherpa.models",
        "sherpa.models.parameter",
    ]
)
