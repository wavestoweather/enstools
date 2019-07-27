# install the ensemble tools
from numpy.distutils.core import setup
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
    with open("enstools/__init__.py") as f:
        for line in f:
            match = re.search('__version__\s*=\s*"([a-zA-Z0-9_.]+)"', line)
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
      packages=['enstools',
                'enstools.core',
                'enstools.scores',
                'enstools.plot',
                'enstools.clustering',
                'enstools.io',
                'enstools.misc',
                'enstools.post',
                'enstools.interpolation',
                'enstools.scores.DisplacementAmplitudeScore',
                'enstools.scores.ScoringRules2Py',
                'enstools.io.eccodes'],
      requires=["numpy",
                "xarray",
                "dask",
                "distributed",
                "cloudpickle",
                "numba",
                "toolz",
                "pint",
                "sklearn",
                "eccodes",
                "cartopy",
                "decorator",
                "multipledispatch",
                "cachey",
                "cffi",
                "pandas"]
      )
