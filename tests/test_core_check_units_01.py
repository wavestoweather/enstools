import xarray
import numpy

from enstools.core import check_arguments


@check_arguments(units={"a": "m", "b": "m", "return_value": "m^2"})
def example_function(a, b):
    """
    Example function for unit checks
    """
    result = a * b
    result.attrs["units"] = "m**2"
    return result


@check_arguments(units={"a": "m", "b": "m", "return_value": "m^2"})
def example_function_no_return_unit(a, b):
    """
    Example function for unit checks. Unit of return value not set
    """
    return a * b


@check_arguments(dims={"a": ["lon", "lat"], "b": ("lon", "lat")})
def example_function_for_dims(a, b):
    """
    Example function for unit checks. Unit of return value not set
    """
    return a * b


@check_arguments(dims={"a": ["lon", "lat"], "b": ("lon", "lat"), "return_value": ("lon", "lat")})
def example_function_for_dims_with_return_value(a, b):
    """
    Example function for unit checks. Unit of return value not set
    """
    return a * b


@check_arguments(dims={"a": ["lon", "lat"], "b": ("lon", "lat"), "return_value": ("lon", "lat")})
def example_function_for_dims_with_wrong_return_value(a, b):
    """
    Example function for unit checks. Unit of return value not set
    """
    return (a * b).sum()


@check_arguments(shape={"a": "b"})
def example_function_with_shape(a, b):
    """
    Example function for unit checks
    """
    result = a * b
    return result


@check_arguments(shape={"a": (2, 3)})
def example_function_with_shape_pre_defined(a, b):
    """
    Example function for unit checks
    """
    result = a * b
    return result


@check_arguments(shape={"a": (2, 3), "return_value": (0, 3)})
def example_function_with_shape_of_return_pre_defined(a, b):
    """
    Example function for unit checks
    """
    result = a * b
    return result


@check_arguments(shape={"a": (0, 0)})
def example_function_keyword(a, b, c=1):
    """
    Example with keyword and default value

    """
    return a * b * c


def test_check_units():
    """
    check the automatic unit check mechanism
    """

    # create example data arrays
    da1 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "km"})
    da2 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "m"})
    da3 = xarray.DataArray(numpy.random.randn(2, 2), attrs={"units": "kg"})
    da4 = xarray.DataArray(numpy.random.randn(2, 2))

    # one array has wrong but convertible units
    res = example_function(da1, da2)
    numpy.testing.assert_array_almost_equal(res, da1 * da2 * 1000.0)

    # one array has wrong not convertible units
    with numpy.testing.assert_raises(ValueError):
        res = example_function(da3, da2)

    # both arguments correct, but return value has no unit
    with numpy.testing.assert_raises(ValueError):
        res = example_function_no_return_unit(da2, da2)

    # test with named arguments
    res = example_function(b=da2, a=da1)


def test_check_dims():
    """
    check the automatic unit check mechanism
    """

    # create example data arrays
    da1 = xarray.DataArray(numpy.random.randn(2, 2), dims=["lon", "lat"])
    da2 = xarray.DataArray(numpy.random.randn(2, 2), dims=["lon", "lat"])
    da3 = xarray.DataArray(numpy.random.randn(2, 2), dims=["lat", "lon"])
    da4 = xarray.DataArray(numpy.random.randn(2, 2), dims=["time", "lon"])

    # both arrays have correct dimensions
    res = example_function_for_dims(da1, da2)

    # one array has wrong order of dimensions
    res = example_function_for_dims(da1, da3)
    numpy.testing.assert_array_equal(res, da3.transpose() * da1)

    # one array has wrong dimensions
    with numpy.testing.assert_raises(ValueError):
        res = example_function_for_dims(da1, da4)

    # one array has wrong order, but order is ignored
    res = example_function_for_dims(da3, da1)

    # check the return value
    res = example_function_for_dims_with_return_value(da3, da2)
    with numpy.testing.assert_raises(ValueError):
        res = example_function_for_dims_with_wrong_return_value(da3, da2)


def test_check_shape():
    """
    check the automatic shape control
    """
    a = numpy.random.randn(2, 2)
    b = numpy.random.randn(3, 2)

    # shape of a should be identical to shape of b
    with numpy.testing.assert_raises(ValueError):
        res = example_function_with_shape(a, b)

    # shape of a should be (2, 3)
    with numpy.testing.assert_raises(ValueError):
        res = example_function_with_shape_pre_defined(a, b)

    # shape of return value should be (?, 3)
    with numpy.testing.assert_raises(ValueError):
        res = example_function_with_shape_of_return_pre_defined(b, b)

    # one value with defined shape, another one without restriction but with default
    res = example_function_keyword(a, b=6)
    numpy.testing.assert_array_equal(res, a*6)
    res = example_function_keyword(a, 6, c=2)
    numpy.testing.assert_array_equal(res, a*12)
