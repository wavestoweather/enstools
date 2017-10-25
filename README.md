# Installation of dependencies in `conda` environment

    conda create -n python2 python=2.7
    conda install -n python2 -c conda-forge numpy numba xarray dask cloudpickle toolz pint nose scikit-learn python-eccodes cartopy decorator multipledispatch
    pip install cachey

# Installation in user home directory

    python setup.py install --user
    
or 

    python3 setup.py install --user

