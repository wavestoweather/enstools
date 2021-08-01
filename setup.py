# install the ensemble tools
from setuptools import setup, find_packages
import re
import sys


def get_version():
    """
    read version string from enstools package without importing it

    Returns
    -------
    str:
            version string
    """
    with open("enstools/core/__init__.py") as f:
        for line in f:
            match = re.search(r'__version__\s*=\s*"([a-zA-Z0-9_.]+)"', line)
            if match is not None:
                return match.group(1)


# only print the version and exit?
if len(sys.argv) == 2 and sys.argv[1] == "--get-version":
    print(get_version())
    exit()


# perform the actual install operation
setup(name="enstools",
      version=get_version(),
      author="Robert Redl et al.",
      author_email="robert.redl@lmu.de",
      namespace_packages=['enstools'],
      packages=find_packages()[1:],
      install_requires=[
                "appdirs",
                "numpy",
                "xarray",
                "dask",
                "distributed",
                "cloudpickle",
                "colorama",
                "numba",
                "toolz",
                "pint",
                "sklearn",
                "cartopy",
                "matplotlib",
                "decorator",
                "multipledispatch",
                "cachey",
                "cffi",
                "bottleneck",
                "pandas",
                "basemap @ git+https://github.com/matplotlib/basemap.git@v1.2.2rel#egg=basemap",
                "h5netcdf",
                "hdf5plugin",
                "zfpy"],
        entry_points={
          'console_scripts': [
              'enstools-opendata=enstools.opendata.cli:main',
              'enstools-compressor=enstools.io:cli'
              ],
        },
      )
