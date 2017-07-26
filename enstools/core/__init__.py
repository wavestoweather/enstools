"""
Core functionality used by other components of the ensemble tools
"""
import sys
import logging
import pint
import inspect
import xarray
from pint.unit import DimensionalityError

# all units from pint
ureg = pint.UnitRegistry()

# default settings
__default_settings = {"check_arguments:convert": True,
                      "check_arguments:strict": True,
                      "check_arguments:reorder": True}


def set_behavior(check_arguments_convert=None, check_arguments_strict=None, check_arguments_reorder=None):
    """
    Change the default behavior of the @check_arguments decorator

    Parameters
    ----------
    check_arguments_convert : bool
            if true, automatic unit conversion takes place

    check_arguments_strict : bool
            if true, missing unit information cause an exception

    """
    if check_arguments_convert is not None:
        __default_settings["check_arguments:convert"] = check_arguments_convert
    if check_arguments_strict is not None:
        __default_settings["check_arguments:strict"] = check_arguments_strict
    if check_arguments_reorder is not None:
        __default_settings["check_arguments:reorder"] = check_arguments_reorder


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


def check_arguments(units={}, dims={}):
    """
    Parameters
    ----------
    units : dict
            Definition of the argument units in the format {"argument_name": "unit"}

    dims : dict
            Definition of the argument dimensions in the format {"argument_name": ("time","lon","lat")} for an exact
            ordering of the dimensions or {"argument_name": ["time","lon","lat"]} for arbitrary ordering of dimensions.
            In the later case, the number of dimensions is not checked, which means that the actual argument is allowed
            to have more dimensions.

    Returns
    -------

    """
    def check_units_decorator(func):
        def func_wrapper(*args, **kwargs):
            # check the actual values of the functions arguments
            if sys.version_info >= (3, 0):
                arg_spec = inspect.getfullargspec(func)
            else:
                arg_spec = inspect.getargspec(func)
            arg_names = arg_spec[0]

            # loop over all none-keyword arguments
            for iarg, one_arg_name in enumerate(arg_names):

                # check the units
                if one_arg_name in units:
                    # is there a units attribute on the variable? Only xarray.DataArrays can have one
                    if isinstance(args[iarg], xarray.DataArray) and "units" in args[iarg].attrs:
                        actual_arg_unit = ureg(args[iarg].attrs["units"])

                        # construct the unit for this argument
                        target_arg_unit = ureg(units[one_arg_name])

                        # the units differ? try to find a conversion!
                        if target_arg_unit != actual_arg_unit:
                            if __default_settings["check_arguments:convert"]:
                                try:
                                    factor = actual_arg_unit.to(target_arg_unit)
                                    args = __replace_argument(args, iarg, args[iarg] * factor.magnitude)
                                    logging.warning("The unit of the argument '%s' was converted from '%s' to '%s' by multiplication with the factor %f" % (arg_names[iarg], actual_arg_unit, target_arg_unit, factor.magnitude))
                                except DimensionalityError as ex:
                                    ex.extra_msg = "; Unable to convert units of argument '%s' of method '%s'!" % (arg_names[iarg], func.__name__)
                                    raise
                            else:
                                logging.warning("The unit of the argument '%s' differs from '%s', no conversion was done!" % (arg_names[iarg], target_arg_unit))


                    # no units attribute, is that an error?
                    else:
                        # construct error or warning message
                        msg = "Argument '%s' has no unit information in form of the 'units' attribute" % arg_names[iarg]
                        if __default_settings["check_arguments:convert"] or __default_settings["check_arguments:strict"]:
                            raise ValueError(msg)
                        else:
                            logging.warning(msg)

                # check the dimensions of the argument
                if one_arg_name in dims:
                    # is it an xarray?
                    if isinstance(args[iarg], xarray.DataArray):
                        actual_arg_dims = args[iarg].dims
                        target_arg_dims = dims[one_arg_name]

                        # are the target dimensions exact or are only required dimensions given?
                        if type(target_arg_dims) == tuple:
                            if actual_arg_dims != target_arg_dims:
                                if __default_settings["check_arguments:reorder"]:
                                    # dimensions differ, is it possible to solve the problem by reordering?
                                    if len(actual_arg_dims) == len(target_arg_dims) and sorted(actual_arg_dims) == sorted(target_arg_dims):
                                        args = __replace_argument(args, iarg, args[iarg].transpose(*target_arg_dims))
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
                        msg = "Argument '%s' is no dimension name information!" % arg_names[iarg]
                        if __default_settings["check_arguments:reorder"] or __default_settings["check_arguments:strict"]:
                            raise ValueError(msg)
                        else:
                            logging.warning(msg)

            # perform the actual calculation
            return_value = func(*args, **kwargs)

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

            # finally return the result
            return return_value

        return func_wrapper
    return check_units_decorator
