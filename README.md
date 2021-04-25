# Installation using pip in local environment (recommended)

At first create a new python virtual environment:

    python3 -m venv --prompt=enstools venv

That will create a new folder `venv` containing the new environment. Next we
need to update `pip` and install `wheel`. Both are required in up-to-date 
versions for the installation to run:

    pip install --upgrade pip wheel

Now we can install `enstools` in development mode into our new environment:

    pip install -e .

# Installation of dependencies in `conda` environment (currently not supported)

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
