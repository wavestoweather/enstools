from enstools.interpolation import unstagger
import numpy as np
import xarray


def test_unstagger_dataarray():
    """
    test of unstagger from enstools.interpolation with a DataArray
    """
    # generate some test data
    rlon = np.arange(10)
    rlat = np.arange(10)
    srlon = np.arange(10) + 0.5
    U = np.random.rand(10, 10)
    da = xarray.DataArray(U, coords={"rlat": rlat, "srlon": srlon}, dims=("rlat", "srlon"))

    # unstagger the grid
    ds2 = unstagger(da)

    # compare values
    np.testing.assert_equal(ds2[0, 0], (da[0, 0] + da[0, 1]) / 2.0)
    np.testing.assert_array_equal(ds2.rlon, rlon[1:])


def test_unstagger_dataset():
    """
    test of unstagger from enstools.interpolation with a Dataset
    """
    # generate some test data
    rlon = np.arange(10)
    rlat = np.arange(10)
    srlon = np.arange(10) + 0.5
    U = np.random.rand(10, 10)
    da = xarray.DataArray(U, coords={"rlat": rlat, "srlon": srlon}, dims=("rlat", "srlon"))
    ds = xarray.Dataset({"U": da})
    ds.coords["rlon"] = rlon

    # unstagger the grid
    ds2 = unstagger(ds)

    # compare values
    np.testing.assert_equal(ds2["U"][0, 0].values, np.nan)
    np.testing.assert_equal(ds2["U"][0, 1], (ds["U"][0, 0] + ds["U"][0, 1]) / 2.0)
