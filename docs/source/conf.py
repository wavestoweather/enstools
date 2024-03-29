# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'enstools'
copyright = '2022'
author = 'Robert Redl'


def get_version():
    from pathlib import Path
    version_path = Path(__file__).parent.parent.parent / "VERSION"
    with version_path.open() as version_file:
        return version_file.read().strip()


version = get_version()
release = version

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'numpydoc',
    'sphinxcontrib.autoprogram',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

