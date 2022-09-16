"""
Core functionality used by other components of the ensemble tools
"""
import six
import sys
import re
import logging
import pint
import inspect
import xarray
import numpy
import dask.array
import dask.multiprocessing
import distributed
import string
from decorator import decorator
from pint import DimensionalityError
from .cluster import init_cluster, get_num_available_procs, get_client_and_worker, all_workers_are_local, \
    RoundRobinWorkerIterator
from .os_support import getstatusoutput, get_cache_dir

# to convert enstools into a namespace package, the version is now listed here and not in the level above
__version__ = "2022.9.1"


class UnitRegistry(pint.UnitRegistry):
    # workaround for Pint issue: https://github.com/hgrecco/pint/issues/476
    def __getattr__(self, name):
        if name[0] == '_':
            try:
                value = super(UnitRegistry, self).__getattr__(name)
                return value
            except pint.errors.UndefinedUnitError as e:
                raise AttributeError()
        else:
            return super(UnitRegistry, self).__getattr__(name)

    # adapted parser for units with minus sign
    def __call__(self, *args, **kwargs):
        return super(UnitRegistry, self).__call__(re.sub("([a-zA-Z]+)(-[0-9]+)", r"\g<1>**\g<2>", args[0]))


# do not use the cache with dask 0.18.0: https://github.com/dask/dask/pull/3632
if dask.__version__ != "0.18.0":
    from dask.cache import Cache
    cache = Cache(2e9)
    cache.register()

# all units from pint
ureg = UnitRegistry()
# add specific units
ureg.define("degrees_east = deg = degree_east = degree_E = degrees_E = degreeE = degreesE.")
ureg.define("degrees_north = deg = degree_north = degree_N = degrees_N = degreeN = degreesN")

# default settings
__default_settings = {"check_arguments:convert": True,
                      "check_arguments:strict": False,
                      "check_arguments:reorder": True}

# default style for logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def set_behavior(check_arguments_convert=None, check_arguments_strict=None, check_arguments_reorder=None, log_level=None):
    """
    Change the default behavior of the @check_arguments decorator

    Parameters
    ----------
    check_arguments_convert : bool
            if true, automatic unit conversion takes place

    check_arguments_strict : bool
            if true, missing unit or dimension information cause an exception

    """
    if check_arguments_convert is not None:
        __default_settings["check_arguments:convert"] = check_arguments_convert
    if check_arguments_strict is not None:
        __default_settings["check_arguments:strict"] = check_arguments_strict
    if check_arguments_reorder is not None:
        __default_settings["check_arguments:reorder"] = check_arguments_reorder
    if log_level is not None:
        if log_level not in ["ERROR", "WARN", "DEBUG", "INFO"]:
            raise ValueError("unsupported log level: '%s'" % log_level)
        __set_log_level(log_level)
        # set the log level also on all workers
        try:
            client = distributed.get_client()
            client.run(__set_log_level, log_level)
        except:
            pass


def __set_log_level(log_level):
    """
    a function executable on all worker processes

    Parameters
    ----------
    level : str
            new log level
    """
    logging.getLogger().setLevel(log_level)


def __replace_argument(args, index, new_arg):
    """
    Replace one argument from within an argument tuple

    Parameters
    ----------
    args : tuple
            tuple of arguments
    index : int
            position of the argument within the tuple
    new_arg
            value of the new argument

    Returns
    -------
    tuple
            new tuple of arguments
    """
    arg_list = list(args)
    arg_list[index] = new_arg
    return tuple(arg_list)


def __shapes_are_equal(shape1, shape2, named_dim_length):
    """
    compare the shape of two array. zeros in the second arguments are ignored

    Parameters
    ----------
    shape1 : tuple
    shape2 : tuple
    named_dim_length : dict

    Returns
    -------
    bool
            True, if both shapes are identical
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape2)):
        if type(shape2[i]) == str:
            # this and all following dimensions have to match
            if shape2[i] in named_dim_length:
                shape2_entry = named_dim_length[shape2[i]]
            else:
                if shape2[i].endswith(":"):
                    named_dim_length[shape2[i]] = shape1[i:]
                else:
                    named_dim_length[shape2[i]] = shape1[i]
                continue
        else:
            shape2_entry = shape2[i]
        if type(shape2_entry) == tuple:
            if shape1[i:] != shape2_entry:
                return False
        elif shape1[i] != shape2_entry and shape2_entry != 0:
            return False
    return True


def get_arg_spec(func):
    """
    returns the arguments of a function and works for python 2 and 3.

    Parameters
    ----------
    func: callable
            function object to inspect
    """
    if six.PY3:
        arg_spec = inspect.getfullargspec(func)
    else:
        arg_spec = inspect.getargspec(func)
    return arg_spec


def check_arguments(units={}, dims={}, shape={}):
    """
    Parameters
    ----------
    units : dict
            Definition of the argument units in the format {"argument_name|argument_index": "unit"}

    dims : dict
            Definition of the argument dimensions in the format {"argument_name|argument_index": ("time","lon","lat")}
            for an exact ordering of the dimensions or {"argument_name|argument_index": ["time","lon","lat"]} for
            arbitrary ordering of dimensions. In the later case, the number of dimensions is not checked, which means
            that the actual argument is allowed to have more dimensions as long as the mentioned dimensions are present.

    shape: dict
            Definition of the arguments shape in the format {"argument_name|argument_index": (10,17)} for a fixed shape
            or {"argument_name|argument_index": "other_argument_name"} if two arguments are required to have the same
            shape. In the first case, zero may be used to indicate, that the size in a direction does not matter, a
            letter may be used if two arguments should have the same shape in this direction. A colon my be added to the
            letter to indicate, that all following dimension have to be identical.

    Returns
    -------

    """

    @decorator
    def check_arguments_decorator(func, *args, **kwargs):
        # check the actual values of the functions arguments
        arg_spec = get_arg_spec(func)
        arg_names = arg_spec[0]
        arg_values = inspect.getcallargs(func, *args, **kwargs)
        # create a list of the new arguments
        modified_args = []
        # dictionary for named dimension length.
        named_dim_length = {}

        # loop over all none-keyword arguments
        for iarg, one_arg_name in enumerate(arg_names):
            current_argument = arg_values[one_arg_name]

            # check the units
            if one_arg_name in units or iarg in units:
                # construct the unit for this argument
                if iarg in units:
                    target_arg_unit = ureg(units[iarg])
                else:
                    target_arg_unit = ureg(units[one_arg_name])

                # is there a units attribute on the variable? Only xarray.DataArrays can have one
                if isinstance(current_argument, xarray.DataArray) and "units" in current_argument.attrs:
                    actual_arg_unit = ureg(current_argument.attrs["units"])

                    # the units differ? try to find a conversion!
                    if target_arg_unit != actual_arg_unit:
                        if __default_settings["check_arguments:convert"]:
                            conversion_failed = False
                            try:
                                factor = actual_arg_unit.to(target_arg_unit)
                                current_argument = current_argument * factor.magnitude
                                logging.warning("The unit of the argument '%s' was converted from '%s' to '%s' by multiplication with the factor %f" % (arg_names[iarg], actual_arg_unit, target_arg_unit, factor.magnitude))
                            except DimensionalityError as ex:
                                conversion_failed = True
                                error_msg = "%s; Unable to convert units of argument '%s' of method '%s'!" % (str(ex), arg_names[iarg], func.__name__)
                            if conversion_failed:
                                raise ValueError(error_msg)
                        else:
                            logging.warning("The unit of the argument '%s' differs from '%s', no conversion was done!" % (arg_names[iarg], target_arg_unit))

                # no units attribute, is that an error?
                else:
                    # construct error or warning message
                    msg = "Argument '%s' has no unit information in form of the 'units' attribute" % arg_names[iarg]
                    if __default_settings["check_arguments:convert"] and __default_settings["check_arguments:strict"]:
                        raise ValueError(msg)
                    else:
                        logging.warning(msg + ", assuming '%s'" % target_arg_unit)

            # check the dimensions of the argument
            if one_arg_name in dims or iarg in dims:
                # is it an xarray?
                if isinstance(current_argument, xarray.DataArray):
                    actual_arg_dims = current_argument.dims
                    if iarg in dims:
                        target_arg_dims = dims[iarg]
                    else:
                        target_arg_dims = dims[one_arg_name]

                    # are the target dimensions exact or are only required dimensions given?
                    if type(target_arg_dims) == tuple:
                        if actual_arg_dims != target_arg_dims:
                            if __default_settings["check_arguments:reorder"]:
                                # dimensions differ, is it possible to solve the problem by reordering?
                                if len(actual_arg_dims) == len(target_arg_dims) and sorted(actual_arg_dims) == sorted(target_arg_dims):
                                    current_argument = current_argument.transpose(*target_arg_dims)
                                    logging.warning("The dimensions of the argument '%s' were reordered from %s to %s" % (arg_names[iarg], actual_arg_dims, target_arg_dims))
                                else:
                                    raise ValueError("unable to change dimensions %s automatically to %s" % (actual_arg_dims, target_arg_dims))
                            else:
                                logging.warning(
                                    "The dimensions of the argument '%s' differs from %s, no reordering was done!" % (
                                    arg_names[iarg], target_arg_dims))

                    # dims are given, but their ordering does not matter
                    elif type(target_arg_dims) == list:
                        for one_dim in target_arg_dims:
                            if one_dim not in actual_arg_dims:
                                msg = "Argument '%s' has no dimension '%s'" % (arg_names[iarg], one_dim)
                                if __default_settings["check_arguments:strict"]:
                                    raise ValueError(msg)
                                else:
                                    logging.warning(msg)

                # no xarray, unable to check dimensions
                else:
                    msg = "Argument '%s' has no dimension name information!" % arg_names[iarg]
                    if __default_settings["check_arguments:reorder"] and __default_settings["check_arguments:strict"]:
                        raise ValueError(msg)
                    else:
                        logging.warning(msg)

            # check the shape of the argument
            if (one_arg_name in shape or iarg in shape) and hasattr(current_argument, "shape"):
                if iarg in shape:
                    shape_entry = shape[iarg]
                else:
                    shape_entry = shape[one_arg_name]
                # is the shape given as tuple or as name of another variable?
                if type(shape_entry) == tuple:
                    if not __shapes_are_equal(current_argument.shape, shape_entry, named_dim_length):
                        raise ValueError("The shape of the argument '%s', which is %s, differs from the pre-defined shape %s" % (one_arg_name, current_argument.shape, shape_entry))
                # shape should be identical to other variables shape
                else:
                    if shape_entry in arg_names:
                        if hasattr(arg_values[shape_entry], "shape"):
                            target_shape = arg_values[shape_entry].shape
                            if current_argument.shape != target_shape:
                                raise ValueError(
                                    "The shape of the argument '%s', which is %s, differs from the shape %s of the reference variable '%s'!" % (
                                    one_arg_name, current_argument.shape, target_shape, shape_entry))
                        else:
                            raise ValueError("The reference variable '%s' has no shape attribute!" % shape_entry)

            # construct new argument list
            if iarg > len(modified_args) and one_arg_name in kwargs:
                kwargs[one_arg_name] = current_argument
            else:
                modified_args.append(current_argument)
                if one_arg_name in kwargs:
                    del kwargs[one_arg_name]

        # perform the actual calculation
        modified_args = tuple(modified_args)
        return_value = func(*modified_args, **kwargs)

        # check the units and convert if necessary and requested
        if "return_value" in units:
            if isinstance(return_value, xarray.DataArray) and "units" in return_value.attrs:
                # get actual and target units and compare
                actual_return_unit = ureg(return_value.attrs["units"])
                target_return_unit = ureg(units["return_value"])
                if actual_return_unit != target_return_unit:
                    if __default_settings["check_arguments:convert"]:
                        try:
                            factor = actual_return_unit.to(target_return_unit)
                            return_value = return_value * factor.magnitude
                            return_value.attrs["units"] = units["return_value"]
                            logging.warning(
                                "The unit of the return value of method '%s' was converted from '%s' to '%s' by multiplication with the factor %f" % (
                                func.__name__, actual_return_unit, target_return_unit, factor.magnitude))
                        except DimensionalityError as ex:
                            ex.extra_msg = "; Unable to convert units of return value of method '%s'!" % (func.__name__)
                            raise

            # unable to check!
            else:
                if __default_settings["check_arguments:convert"] or __default_settings["check_arguments:strict"]:
                    msg = "Return value of function '%s' has no unit information in form of the 'units' attribute" % func.__name__
                    raise ValueError(msg)

        # check the dimensions and convert if necessary and requested
        if "return_value" in dims:
            # is it an xarray?
            if isinstance(return_value, xarray.DataArray):
                actual_arg_dims = return_value.dims
                target_arg_dims = dims[one_arg_name]

                # are the target dimensions exact or are only required dimensions given?
                if type(target_arg_dims) == tuple:
                    if actual_arg_dims != target_arg_dims:
                        if __default_settings["check_arguments:reorder"]:
                            # dimensions differ, is it possible to solve the problem by reordering?
                            if len(actual_arg_dims) == len(target_arg_dims) and sorted(actual_arg_dims) == sorted(
                                    target_arg_dims):
                                return_value = return_value.transpose(*target_arg_dims)
                                logging.warning(
                                    "The dimensions of the return value were reordered from %s to %s" % (actual_arg_dims, target_arg_dims))
                            else:
                                raise ValueError("unable to change dimensions of return value %s automatically to %s" % (actual_arg_dims, target_arg_dims))
                        else:
                            logging.warning(
                                "The dimensions of the return value differs from %s, no reordering was done!" % (
                                    arg_names[iarg], target_arg_dims))

                # dims are given, but their ordering does not matter
                elif type(target_arg_dims) == list:
                    for one_dim in target_arg_dims:
                        if one_dim not in actual_arg_dims:
                            msg = "The return value has no dimension '%s'" % one_dim
                            if __default_settings["check_arguments:strict"]:
                                raise ValueError(msg)
                            else:
                                logging.warning(msg)

            # no xarray, unable to check dimensions
            else:
                msg = "The return value has no dimension name information!"
                if __default_settings["check_arguments:reorder"] or __default_settings["check_arguments:strict"]:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)

        # check the shape of the return value
        if "return_value" in shape:
            # is the shape given as tuple or as name of another variable?
            if type(shape["return_value"]) == tuple:
                if not __shapes_are_equal(return_value.shape, shape["return_value"], named_dim_length):
                    raise ValueError(
                        "The shape of the return value, which is %s, differs from the pre-defined shape %s" % (
                        return_value.shape, shape["return_value"]))
            # shape should be identical to other variables shape
            else:
                if shape["return_value"] in arg_names:
                    if hasattr(arg_values[shape["return_value"]], "shape"):
                        target_shape = arg_values[shape["return_value"]].shape
                        if return_value.shape != target_shape:
                            raise ValueError(
                                "The shape of the return value, which is %s, differs from the shape %s of the reference variable '%s'!" % (
                                    return_value.shape, target_shape, shape["return_value"]))
                    else:
                        raise ValueError(
                            "The reference variable '%s' has no shape attribute!" % shape["return_value"])

        # finally return the result
        return return_value

    return check_arguments_decorator


def get_chunk_size_for_n_procs(shape, nproc):
    """
    Create a chunk distribution for dask-arrays depending on the number of available processors. Chunks are created
    from the two right-most dimensions.

    Parameters
    ----------
    nproc : int
            number of available processors

    shape : tuple
            shape of the array to split up

    Returns
    -------
    tuple
            chunk-tuple usable in dask arrays

    Examples
    --------
    >>> get_chunk_size_for_n_procs(100, 10)
    (10,)
    >>> get_chunk_size_for_n_procs((100, 100), 8)
    (50, 25)
    >>> get_chunk_size_for_n_procs((1000, 100), 8)
    (125, 100)
    >>> get_chunk_size_for_n_procs((100, 1000), 8)
    (100, 125)
    >>> get_chunk_size_for_n_procs((10, 1000, 1000), 8)
    (10, 500, 250)
    """
    if type(shape) == int:
        shape = (shape,)
    # use the right-most two dims
    if len(shape) <= 2:
        shape_for_chunk = shape
    else:
        shape_for_chunk = shape[-2:]

    # decide for the number of chunks in each direction
    if len(shape_for_chunk) == 1:
        return (shape_for_chunk[0] // nproc,)
    if len(shape_for_chunk) == 2:
        ratio = shape_for_chunk[1] / float(shape_for_chunk[0])
        if 0.5 <= ratio <= 2.0:
            n_chunks_0 = int(numpy.sqrt(nproc))
            n_chunks_1 = nproc // n_chunks_0
        elif ratio < 0.5:
            n_chunks_0 = nproc
            n_chunks_1 = 1
        elif ratio > 2.0:
            n_chunks_0 = 1
            n_chunks_1 = nproc
        if len(shape) > len(shape_for_chunk):
            return shape[:-len(shape_for_chunk)] + (shape_for_chunk[0] // n_chunks_0, shape_for_chunk[1] // n_chunks_1)
        else:
            return shape_for_chunk[0] // n_chunks_0, shape_for_chunk[1] // n_chunks_1


def parallelize_univariate_two_arg(func):
    """
    Parallelize a function with two arguments. The first argument is a (N,...)-array, the second argument is a
    (e,N,...)-array. The array in the first argument will be divided into chunks of (almost) equal size. The array in
    the second argument will be divided in the same way in its rightmost dimensions.

    The first call to the underlining function will be func(arg0[0:chunk0], arg1[:,0:chunk0]),
    the second func(arg0[chunk0:chunk1], arg1[:,chunk0:chunk1]), ...

    Parameters
    ----------
    func : function object

    Returns
    -------
    numpy.ndarray or xarray.DataArray or float
    """

    # create a wrapper for argument and result conversion conversion
    @check_arguments(shape={0: ("n-obs:",), 1: ("ensemble", "n-obs:")})
    def function_wrapper(arg0, arg1, mean=False, **kwargs):
        # paralyze with dask
        # calculate chunk sizes
        nprocs = get_num_available_procs()
        chunk_size = get_chunk_size_for_n_procs(arg0.shape, nprocs)
        da0 = dask.array.from_array(arg0, chunks=chunk_size)
        da1 = dask.array.from_array(arg1, chunks=(arg1.shape[0],) + chunk_size)

        # perform the actual calculation on a chunk of the array
        def dask_calculation(a0, a1):
            if len(a0.shape) > 1:
                original_shape = a0.shape
                a0 = a0.flatten()
                a1 = a1.reshape((a1.size // a0.size, a0.size))
                result = func(a0, numpy.moveaxis(a1, 0, -1), **kwargs)
                result = result.reshape(original_shape)
            else:
                result = func(a0, numpy.moveaxis(a1, 0, -1), **kwargs)
            return result

        # order of index in input and output
        obs_ind = string.ascii_lowercase[:len(chunk_size)]
        fct_int = "z"+obs_ind

        # perform the actual calculation
        result = dask.array.atop(dask_calculation, obs_ind, da0, obs_ind, da1, fct_int, dtype=da0.dtype, concatenate=True)

        # calculate mean if requested
        if mean:
            return result.mean().compute(get=dask.multiprocessing.get, num_workers=nprocs)
        else:
            return result.compute(get=dask.multiprocessing.get, num_workers=nprocs)

    return function_wrapper


def vectorize_univariate_two_arg(func):
    """
    Vectorize a function with two arguments. The first argument if a 1d-array, the second argument is a 2d-array. The
    first call to the underlining function will be func(arg0[0], arg1[:,0]), the second func(arg0[1], arg1[:,1]), ...

    Parameters
    ----------
    func : function object

    Returns
    -------
    numpy.ndarray or xarray.DataArray or float
    """

    # create the vectorized version
    vfunc = numpy.vectorize(func, signature="(),(n)->()")

    # create a wrapper for argument and result conversion conversion
    @check_arguments(shape={0: ("n-obs:",), 1: ("ensemble", "n-obs:")})
    def function_wrapper(arg0, arg1, mean=False, **kwargs):
        # perform the actual calculation
        result = vfunc(arg0, numpy.moveaxis(arg1, 0, -1), **kwargs)

        # calculate mean if requested
        if mean:
            return result.mean()
        else:
            return result

    return function_wrapper


def vectorize_multivariate_two_arg(func, arrays_concatenated=True):
    """
    Vectorize a function with two arguments. The first argument if a 2d-array, the second argument is a 3d-array. The
    first call to the underlining function will be func(arg0[:,0], arg1[:,:,0]), the second
    func(arg0[:,1], arg1[:,:,1]), ...

    Parameters
    ----------
    func : function object
    arrays_concatenated : bool
            if true, the function has one observation and one forecast argument, if false the function has two
            observation and two forecast arguments

    Returns
    -------
    numpy.ndarray or xarray.DataArray or float
    """
    # check the arguments of the functions. only the first two arguments are used in vectorization.
    if sys.version_info >= (3, 0):
        arg_spec = inspect.getfullargspec(func)
    else:
        arg_spec = inspect.getargspec(func)

    # create the vectorized version
    if len(arg_spec.args) > 2:
        vfunc = numpy.vectorize(func, signature="(d),(d,m)->()", excluded=arg_spec.args[2:])
    else:
        vfunc = numpy.vectorize(func, signature="(d),(d,m)->()")

    def __vectorize_multivariate_two_arg_concatenated(arg0, arg1, mean=False, **kwargs):
        # perform the actual calculation
        if type(arg0) == xarray.DataArray:
            arg0 = arg0.data
        if type(arg1) == xarray.DataArray:
            arg1 = arg1.data
        result = vfunc(numpy.moveaxis(arg0, 0, -1), numpy.moveaxis(arg1, (0, 1), (-2, -1)), **kwargs)

        # calculate mean if requested
        if mean:
            return result.mean()
        else:
            return result

    # create a wrapper for argument and result conversion conversion
    if arrays_concatenated:
        return check_arguments(shape={0: ("obs-dim", "n-obs:"), 1: ("obs-dim", "ensemble", "n-obs:")})(__vectorize_multivariate_two_arg_concatenated)

    else:
        @check_arguments(shape={0: ("n-obs:",), 1: ("n-obs:",), 2: ("ensemble", "n-obs:"), 3: ("ensemble", "n-obs:")})
        def function_wrapper(obs0, obs1, fct0, fct1, mean=False, **kwargs):
            # concatenate arrays
            arg0 = numpy.stack((obs0, obs1))
            arg1 = numpy.stack((fct0, fct1))

            # call the version of this function for concatenated arrays
            return __vectorize_multivariate_two_arg_concatenated(arg0, arg1, mean, **kwargs)

        return function_wrapper


def import_multipledispatch(dispatcher, globals):
    """
    Import all implementations of a function created by the multipledispatch module into the globals dictionary.
    They will be made available by name_md%02d, where *%02d* is a counter.

    The purpose of this function is to provide compatibility for multiple dispatch functions with sphinx.

    Parameters
    ----------
    dispatcher
            the dispatcher function object

    globals
            namespace into which the function and all its implementations should be imported

    """
    # iterate over all implementations
    imported = set()
    i = 0
    for signature in sorted(dispatcher.funcs.keys(), key=lambda x: str(x)):
        func = dispatcher.funcs[signature]
        if not func in imported:
            i += 1
            globals["%s_md%02d" % (dispatcher.__name__, i)] = func
            imported.add(func)
