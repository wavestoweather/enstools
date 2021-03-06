# -*- coding: utf-8 -*-
#
# Ensemble Tools documentation build configuration file, created by
# sphinx-quickstart on Wed Aug  9 17:36:52 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import re
import os
import random
import string


def get_version(short=False):
    """
    read version string from enstools package without importing it

    Returns
    -------
    str:
            version string
    """
    with open("../enstools/__init__.py") as f:
        for line in f:
            match = re.search('__version__\s*=\s*"([a-zA-Z0-9_.]+)"', line)
            if match is not None:
                if short:
                    match = re.search('([0-9]+\.[0-9]+)', match.group(1))
                return match.group(1)


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
autoclass_content = 'both'

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.extlinks',
    'sphinx.ext.autosummary',
    #'sphinx.ext.napoleon',
    'numpydoc'
]

#numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi:'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Ensemble Tools'
copyright = u'2017, Waves to Weather, a Transregional Collaborative Research Center'
author = u'Redl et al., Contributors from Waves to Weather'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = get_version(short=True)
# The full version, including alpha/beta/rc tags.
release = get_version()

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'EnsembleToolsdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'EnsembleTools.tex', u'Ensemble Tools Documentation',
     u'Robert Redl et al.', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ensembletools', u'Ensemble Tools Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'EnsembleTools', u'Ensemble Tools Documentation',
     author, 'EnsembleTools', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

# Settings for automodule
import sphinx.ext.autodoc as autodoc
from sphinx.util.inspect import getargspec, is_builtin_class_method
import inspect
import multipledispatch
import warnings

# disable warning from numpydoc about Dispatcher docstrings
warnings.filterwarnings("ignore", message="^Unknown section Inputs: <")


class DispatcherDocumenter(autodoc.FunctionDocumenter):
    """
    Specialized Documenter subclass for multiple dispatch functions.
    """
    objtype = 'function'
    member_order = 30

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, (autodoc.FunctionType, multipledispatch.Dispatcher))

    def get_doc(self, encoding=None, ignore=1):
        if not isinstance(self.object, multipledispatch.Dispatcher):
            return super(DispatcherDocumenter, self).get_doc(encoding, ignore)
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines
        # iterate over all implementations
        implementations = set()
        lines = []
        i = 0
        for signature in sorted(self.object.funcs.keys(), key=lambda x: str(x)):
            func = self.object.funcs[signature]
            if not func in implementations:
                i += 1
                implementations.add(func)
                docstring = self.get_attr(func, '__doc__', None)
                # add signature
                lines.append([u"``%s%s``" % (self.object_name, self.format_args_one_func(func))])
                # make sure we have Unicode docstrings, then sanitize and split
                # into lines
                if isinstance(docstring, autodoc.text_type):
                    lines.extend([autodoc.prepare_docstring(docstring, ignore)])
                elif isinstance(docstring, str):  # this will not trigger on Py3
                    lines.extend([autodoc.prepare_docstring(autodoc.force_decode(docstring, encoding), ignore)])
        return lines

    def format_args(self):
        if not isinstance(self.object, multipledispatch.Dispatcher):
            return super(DispatcherDocumenter, self).format_args()

        # iterate over all implementations
        implementations = set()
        arg_list = []
        i = 0
        for signature in sorted(self.object.funcs.keys(), key=lambda x: str(x)):
            func = self.object.funcs[signature]
            if not func in implementations:
                i += 1
                implementations.add(func)
                arg_list.append(self.format_args_one_func(func))
        return " or ".join(arg_list)

    def format_signature(self):
        """
        format the signatures of this multiple dispatcher function

        Returns
        -------
        """
        if not isinstance(self.object, multipledispatch.Dispatcher):
            return super(DispatcherDocumenter, self).format_signature()
        return ''

    @staticmethod
    def format_args_one_func(func):
        """
        returns the docstring for one function

        Parameters
        ----------
        func : function
                function object with docstring
        """
        if inspect.isbuiltin(func) or \
                inspect.ismethoddescriptor(func):
            # cannot introspect arguments of a C function or method
            return None
        try:
            argspec = getargspec(func)
        except TypeError:
            if (is_builtin_class_method(func, '__new__') and
                    is_builtin_class_method(func, '__init__')):
                raise TypeError('%r is a builtin class' % func)

            # if a class should be documented as function (yay duck
            # typing) we try to use the constructor signature as function
            # signature without the first argument.
            try:
                argspec = getargspec(func.__new__)
            except TypeError:
                argspec = getargspec(func.__init__)
                if argspec[0]:
                    del argspec[0][0]
        args = autodoc.formatargspec(function=func, args=argspec.args, varargs=argspec.varargs, varkw=argspec.varkw,
                                     defaults=argspec.defaults, kwonlyargs=argspec.kwonlyargs,
                                     kwonlydefaults=argspec.kwonlydefaults, annotations=argspec.annotations)
        # escape backslashes for reST
        args = args.replace('\\', '\\\\')
        return args


def setup(app):
    app.add_autodocumenter(DispatcherDocumenter)
