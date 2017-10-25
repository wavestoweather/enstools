# install the ensemble tools
from numpy.distutils.core import setup

setup(name="enstools",
      version="0.0.1",
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
                "cloudpickle",
                "numba",
                "toolz",
                "pint",
                "sklearn",
                "eccodes",
                "cartopy",
                "decorator",
                "multipledispatch"
                "cachey"]
      )
