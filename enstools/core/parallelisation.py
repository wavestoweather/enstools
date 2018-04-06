"""
helper functions for automatic parallelization using dask
"""
from decorator import decorate, decorator
from enstools.misc import get_time_dim, get_ensemble_dim
import xarray
import dask
from distributed import wait
import numpy as np
import logging
import six
from enstools.core import get_arg_spec
import inspect


def __args_to_dask(*args):
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
        # how should be re-chunked?
        if isinstance(one_arg, dask.array.core.Array):
            new_args.append(one_arg)
        elif isinstance(one_arg, xarray.core.dataset.Dataset):
            # it is a dataset, check all data variables individually
            # first pass: are there any none-dask arrays?
            has_none_dask = False
            for one_data_var_name, one_data_var in six.iteritems(one_arg.data_vars):
                if not isinstance(one_data_var, dask.array.core.Array):
                    has_none_dask = True
                    break
            # we have none-dask arrays, create a copy one the dataset and replace single variables
            if has_none_dask:
                new_one_arg = one_arg.copy()
                for one_data_var_name, one_data_var in six.iteritems(one_arg.data_vars):
                    if not isinstance(one_data_var.data, dask.array.core.Array):
                        new_one_data_var = __args_to_dask(one_data_var)[0]
                        new_one_arg[one_data_var_name].data = new_one_data_var
                new_args.append(new_one_arg)
            else:
                new_args.append(one_arg)
        elif isinstance(one_arg, xarray.core.dataarray.DataArray):
            if isinstance(one_arg.data, dask.array.core.Array):
                # if the data is already a dask array, wait for it to become ready
                try:
                    wait(one_arg.data)
                except ValueError:
                    logging.debug("not client found, can not wait for computations on the cluster!")
                new_args.append(one_arg.data)
            else:
                new_args.append(one_arg.chunk().data)
        elif isinstance(one_arg, np.ndarray):
            new_args.append(dask.array.from_array(one_arg, chunks=-1))
        else:
            new_args.append(one_arg)
    return new_args


def rechunk_arguments(dim_type=None, dim_name=None, arg_names=None):
    """
    rechunk all arguments along a specific dimensions

    Parameters
    ----------
    arg : list
            list of arguments. all of them must be xarray DataArrays or Datasets

    dim_type : dict
            keys: dimensions types
            "time": all known names for time dimensions are used
            "ens": all known names for ensemble dimensions are used
            values: chunk size along this dimension

    dim_name : dict
            use exactly this dimension name, do not search for dimensions of this type with other names.

    arg_names : list
            list of arguments to process. None: all arguments without defaults. This argument is not yet implemented!
    """
    # check decorator arguments
    if dim_type is None and dim_name is None:
        raise ValueError("the decorator rechunk_arguments needs at least one of the arguments dim_type or dim_name!")
    if dim_type is None:
        dim_type = {}
    if dim_name is None:
        dim_name = {}

    @decorator
    def function_wrapper(func, *args, **kwargs):
        # check the actual values of the functions arguments
        arg_spec = get_arg_spec(func)
        arg_names = arg_spec[0]
        defaults = arg_spec[3]
        arg_values = inspect.getcallargs(func, *args, **kwargs)

        # create a list of the new arguments
        new_args = []

        # loop over all none-keyword arguments
        for iarg, one_arg_name in enumerate(arg_names):
            one_arg = arg_values[one_arg_name]
            if iarg >= len(arg_names) - len(defaults):
                pass
            elif isinstance(one_arg, xarray.core.dataarray.DataArray) or isinstance(one_arg, xarray.core.dataset.Dataset):
                chunks = {}
                # find the dimension to re-chunk
                if "time" in dim_type:
                    dn = get_time_dim(one_arg)
                    chunks[dn] = dim_type["time"]
                if "ens" in dim_type:
                    dn = get_ensemble_dim(one_arg)
                    chunks[dn] = dim_type["ens"]
                # loop over all explicitly named dimensions
                for dn, cs in six.iteritems(dim_name):
                    if dn in one_arg:
                        chunks[dn] = cs
                # apply the re-chunk operation
                one_arg = one_arg.chunk(chunks)
            # allow scalar arguments
            elif isinstance(one_arg, int) or isinstance(one_arg, float):
                pass
            else:
                raise ValueError("automatic re-chunking is only possible with xarray arguments!")

            # construct new argument list
            if iarg > len(new_args) and one_arg_name in kwargs:
                kwargs[one_arg_name] = one_arg
            else:
                new_args.append(one_arg)
                if one_arg_name in kwargs:
                    del kwargs[one_arg_name]

        return func(*new_args, **kwargs)

    return function_wrapper


def chunkwise(func):
    """
    Automatic parallelisation of a decorated function. All arguments have to have the same shape!
    """

    def function_wrapper(func, *args, **kwargs):
        dask_args = __args_to_dask(*args)
        result = dask.array.map_blocks(func, *dask_args, dtype=args[0].dtype)
        return result

    return decorate(func, function_wrapper)


def timestepwise(rechunk_only=False):
    """
    Automatic parallelisation of a decorated function. All arguments have to have the same shape!
    """
    @decorator
    def function_wrapper(func, *args, **kwargs):
        dask_args = __args_to_dask(*args)
        result = dask.array.map_blocks(func, *dask_args, dtype=args[0].dtype)
        return result

    return function_wrapper
