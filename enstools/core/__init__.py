"""
Core functionality used by other components of the ensemble tools
"""
import sys
import re
import logging
import pint
import inspect
import xarray
from functools import wraps
from pint import DimensionalityError


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
        return super(UnitRegistry, self).__call__(re.sub("([a-zA-Z]+)(-[0-9]+)", "\g<1>**\g<2>", args[0]))


# all units from pint
ureg = UnitRegistry()

# default settings
__default_settings = {"check_arguments:convert": True,
                      "check_arguments:strict": True,
                      "check_arguments:reorder": True}

# default style for logging
logging.basicConfig(level=logging.WARN, format="%(levelname)s: %(message)s")


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


def check_arguments(units={}, dims={}, shape={}):
    """
    Parameters
    ----------
    units : dict
            Definition of the argument units in the format {"argument_name": "unit"}

    dims : dict
            Definition of the argument dimensions in the format {"argument_name": ("time","lon","lat")} for an exact
            ordering of the dimensions or {"argument_name": ["time","lon","lat"]} for arbitrary ordering of dimensions.
            In the later case, the number of dimensions is not checked, which means that the actual argument is allowed
            to have more dimensions as long as the mentioned dimensions are present.

    shape: dict
            Definition of the arguments shape in the format {"argument_name": (10,17)} for a fixed shape or
            {"argument_name": "other_argument_name"} if two arguments are required to have the same shape.

    Returns
    -------

    """
    def check_units_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            # check the actual values of the functions arguments
            if sys.version_info >= (3, 0):
                arg_spec = inspect.getfullargspec(func)
            else:
                arg_spec = inspect.getargspec(func)
            arg_names = arg_spec[0]
            arg_values = inspect.getcallargs(func, *args, **kwargs)
            # create a list of the new arguments
            modified_args = []


            # loop over all none-keyword arguments
            for iarg, one_arg_name in enumerate(arg_names):
                current_argument = arg_values[one_arg_name]

                # check the units
                if one_arg_name in units:
                    # is there a units attribute on the variable? Only xarray.DataArrays can have one
                    if isinstance(current_argument, xarray.DataArray) and "units" in current_argument.attrs:
                        actual_arg_unit = ureg(current_argument.attrs["units"])

                        # construct the unit for this argument
                        target_arg_unit = ureg(units[one_arg_name])

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
                        if __default_settings["check_arguments:convert"] or __default_settings["check_arguments:strict"]:
                            raise ValueError(msg)
                        else:
                            logging.warning(msg)

                # check the dimensions of the argument
                if one_arg_name in dims:
                    # is it an xarray?
                    if isinstance(current_argument, xarray.DataArray):
                        actual_arg_dims = current_argument.dims
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
                        msg = "Argument '%s' is no dimension name information!" % arg_names[iarg]
                        if __default_settings["check_arguments:reorder"] or __default_settings["check_arguments:strict"]:
                            raise ValueError(msg)
                        else:
                            logging.warning(msg)

                # check the shape of the argument
                if one_arg_name in shape and hasattr(current_argument, "shape"):
                    # is the shape given as tuple or as name of another variable?
                    if type(shape[one_arg_name]) == tuple:
                        if current_argument.shape != shape[one_arg_name]:
                            raise ValueError("The shape of the argument '%s', which is %s, differs from the pre-defined shape %s" % (one_arg_name, current_argument.shape, shape[one_arg_name]))
                    # shape should be identical to other variables shape
                    else:
                        if shape[one_arg_name] in arg_names:
                            if hasattr(arg_values[shape[one_arg_name]], "shape"):
                                target_shape = arg_values[shape[one_arg_name]].shape
                                if current_argument.shape != target_shape:
                                    raise ValueError(
                                        "The shape of the argument '%s', which is %s, differs from the shape %s of the reference variable '%s'!" % (
                                        one_arg_name, current_argument.shape, target_shape, shape[one_arg_name]))
                            else:
                                raise ValueError("The reference variable '%s' has no shape attribute!" % shape[one_arg_name])

                # construct new argument list
                if one_arg_name in kwargs:
                    kwargs[one_arg_name] = current_argument
                else:
                    modified_args.append(current_argument)

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

            # finally return the result
            return return_value

        return func_wrapper
    return check_units_decorator
