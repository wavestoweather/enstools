"""
Core functionality used by other components of the ensemble tools
"""
import logging
import pint
import inspect
import xarray
from pint.unit import DimensionalityError

# all units from pint
ureg = pint.UnitRegistry()

# default settings
__default_settings = {"check_units:convert": True,
                      "check_units:strict": True}


def set_behavior(check_units_convert=None, check_units_strict=None):
    """
    Change the default behavior of the @check_units decorator

    Parameters
    ----------
    check_units_convert : bool
            if true, automatic unit conversion takes place

    check_units_strict : bool
            if true, missing unit information cause an exception

    """
    if check_units_convert is not None:
        __default_settings["check_units:convert"] = check_units_convert
    if check_units_strict is not None:
        __default_settings["check_units:strict"] = check_units_strict


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


def check_units(**unitdefinitions):
    """

    Parameters
    ----------
    convert : bool
            if true, automatic unit conversion takes place.

    strict : bool
            missing 'units' attributes are considered to be errors

    **unitdefinitions
            Definition of the target units.

    Returns
    -------

    """
    def check_units_decorator(func):
        def func_wrapper(*args, **kwargs):
            # check the actual values of the functions arguments
            arg_spec = inspect.getfullargspec(func)
            arg_names = arg_spec[0]

            # loop over all none-keyword arguments
            for iarg, one_arg_name in enumerate(arg_names):
                if one_arg_name in unitdefinitions:
                    # is there a units attribute on the variable? Only xarray.DataArrays can have one
                    if isinstance(args[iarg], xarray.DataArray) and "units" in args[iarg].attrs:
                        actual_arg_unit = ureg(args[iarg].attrs["units"])

                        # construct the unit for this argument
                        target_arg_unit = ureg(unitdefinitions[one_arg_name])

                        # the units differ? try to find a conversion!
                        if target_arg_unit != actual_arg_unit:
                            if __default_settings["check_units:convert"]:
                                try:
                                    factor = actual_arg_unit.to(target_arg_unit)
                                    args = __replace_argument(args, iarg, args[iarg] * factor.magnitude)
                                    logging.warning("The unit of the argument '%s' was converted from '%s' to '%s' by multiplication with the factor %f" % (arg_names[iarg], actual_arg_unit, target_arg_unit, factor.magnitude))
                                except DimensionalityError as ex:
                                    ex.extra_msg = "; Unable to convert units of argument '%s' of method '%s'!" % (arg_names[iarg], func.__name__)
                                    raise

                    # no units attribute, is that an error?
                    else:
                        # construct error or warning message
                        msg = "Argument '%s' has no unit information in form of the 'units' attribute" % arg_names[iarg]
                        if __default_settings["check_units:convert"] or __default_settings["check_units:strict"]:
                            raise ValueError(msg)
                        else:
                            logging.warning(msg)

            # perform the actual calculation
            return_value = func(*args, **kwargs)

            # check the units and convert necessary and requested
            if "return_value" in unitdefinitions:
                if isinstance(return_value, xarray.DataArray) and "units" in return_value.attrs:
                    # get actual and target units and compare
                    actual_return_unit = ureg(return_value.attrs["units"])
                    target_return_unit = ureg(unitdefinitions["return_value"])
                    if actual_return_unit != target_return_unit:
                        if __default_settings["check_units:convert"]:
                            try:
                                factor = actual_return_unit.to(target_return_unit)
                                return_value = return_value * factor.magnitude
                                return_value.attrs["units"] = unitdefinitions["return_value"]
                                logging.warning(
                                    "The unit of the return value of method '%s' was converted from '%s' to '%s' by multiplication with the factor %f" % (
                                    func.__name__, actual_return_unit, target_return_unit, factor.magnitude))
                            except DimensionalityError as ex:
                                ex.extra_msg = "; Unable to convert units of return value of method '%s'!" % (func.__name__)
                                raise

                # unable to check!
                else:
                    if __default_settings["check_units:convert"] or __default_settings["check_units:strict"]:
                        msg = "Return value of function '%s' has no unit information in form of the 'units' attribute" % func.__name__
                        raise ValueError(msg)

            # finally return the result
            return return_value

        return func_wrapper
    return check_units_decorator
