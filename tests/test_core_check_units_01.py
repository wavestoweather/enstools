import xarray
import numpy
from pint.unit import DimensionalityError

from enstools.core import check_units


@check_units(a="m", b="m", return_value="m^2")
def example_function(a, b):
    """
    Example function for unit checks
    """
    result = a * b
    result.attrs["units"] = "m**2"
    return result


@check_units(a="m", b="m", return_value="m^2")
def example_function_no_return_unit(a, b):
    """
    Example function for unit checks. Unit of return value not set
    """
    return a * b


def test_check_units():
    """
    check the automatic unit check mechanism
    """

    # create example data arrays
    da1 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "km"})
    da2 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "m"})
    da3 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "kg"})
    da4 = xarray.DataArray(numpy.random.randn(2, 2))

    # one array has wrong but convertable units
    res = example_function(da1, da2)
    numpy.testing.assert_array_almost_equal(res, da1 * da2 * 1000.0)

    # one array has wrong not convertable units
    with numpy.testing.assert_raises(DimensionalityError):
        res = example_function(da3, da2)

    # both arguments correct, but return value has no unit
    with numpy.testing.assert_raises(ValueError):
        res = example_function_no_return_unit(da2, da2)
