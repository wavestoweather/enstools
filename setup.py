# Install the ensemble tools
from setuptools import setup, find_packages
import re
import sys
import os.path

# Use the Readme file as long description.
try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""


def get_version():
    from pathlib import Path
    version_path = Path(__file__).parent / "VERSION"
    with version_path.open() as version_file:
        return version_file.read().strip()


def find_enstools_packages():
    """
    Find the packages inside the enstools folder.
    """

    return [f'enstools.{p}' for p in (find_packages(f'{os.path.dirname(__file__)}/enstools'))]


# only print the version and exit?
if len(sys.argv) == 2 and sys.argv[1] == "--get-version":
    print(get_version())
    exit()

# perform the actual install operation
setup(name="enstools",
      version=get_version(),
      author="Robert Redl et al.",
      author_email="robert.redl@lmu.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/wavestoweather/enstools",
      packages=['enstools', *find_enstools_packages()],
      install_requires=[
          "appdirs",
          "numpy",
          "xarray",
          "cftime",
          "dask",
          "distributed",
          "cloudpickle",
          "colorama",
          "numba",
          "toolz",
          "pint",
          "matplotlib",
          "decorator",
          "multipledispatch",
          "cachey",
          "cffi",
          "pandas",
          "packaging",
          "h5netcdf",
          "h5py",
          "numcodecs",
          "scikit-image",
          "scikit-learn",
          "scipy",
          "statsmodels",
          "dataclasses",
      ],
    extras_require={
        'plot': [
            'cartopy',
            'plotly',
            'bokeh']
        ,

        'compression': ['enstools-compression',]
        ,

        # you can define other optional groups of dependencies here
    },
      entry_points={
          'console_scripts': [
              'enstools-opendata=enstools.opendata.cli:main',
          ],
      },
      )
