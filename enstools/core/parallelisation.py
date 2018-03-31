"""
helper functions for automatic parallelization using dask
"""
from decorator import decorate
import xarray
import dask
import numpy as np


def args_to_dask(*args):
    """
    convert arguments to dask arrays or extract dask arrays if already present.

    Parameters
    ----------
    args :
            numpy, xarray, or dask array

    Returns
    -------
    args :
            all arguments converted to dask
    """
    new_args = []
    for one_arg in args:
        if isinstance(one_arg, dask.array.core.Array):
            new_args.append(one_arg)
        elif isinstance(one_arg, xarray.core.dataarray.DataArray):
            if isinstance(one_arg.data, dask.array.core.Array):
                new_args.append(one_arg.data)
            else:
                new_args.append(one_arg.chunk().data)
        elif isinstance(one_arg, np.ndarray):
            new_args.append(dask.array.from_array(one_arg, chunks=-1))
        else:
            new_args.append(one_arg)
    return new_args


def chunkwise(func):
    """
    Automatic parallelisation of a decorated function. All arguments have to have the same shape!
    """

    def function_wrapper(func, *args, **kwargs):
        #result = xarray.apply_ufunc(func, *args, dask="allowed", **kwargs)
        dask_args = args_to_dask(*args)
        result = dask.array.map_blocks(func, *dask_args, dtype=args[0].dtype)
        return result

    return decorate(func, function_wrapper)
