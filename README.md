# Installation of dependencies in `conda` environment

The `enstools` package has now full support for Python 2 and Python 3. For new developments, it is highly recommended to
use only Python 3. Important dependencies like `numpy` will stop to support Python 2 soon.

    conda create -n enstools
    conda install -n enstools -c conda-forge numpy numba xarray dask distributed cloudpickle toolz pint nose scikit-learn eccodes cartopy decorator multipledispatch cffi cachey
    source activate enstools

# Installation in user home directory

    python2 setup.py install --user
    
or 

    python3 setup.py install --user

If you want to use the `crps` function you have to install it in R once manually:

1.) open R
2.) install.packages("scoringRules")
